"""Train pi-GAN. Supports distributed training."""
import argparse
import os
import os.path as osp
import math
import copy
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

# from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from kotools import PLog

from generators import generators
from discriminators import discriminators
from siren import siren
import fid_evaluation

import datasets
import curriculums

# pylint: disable=no-member
safelog10 = lambda x: 0.0 if not x else np.log10(np.abs(x))
sround = lambda x, d=1: np.round(x, max((-np.floor(safelog10(x)).astype(int) + d), 0))

def _continue(folder, init=False):
    """
    Create place holder file. Kill file to stop training at end of epoch
    """
    _cont = osp.join(folder, "_continue_training")
    if init and not osp.isfile(_cont):
        with open(_cont, "w") as _fi:
            pass
    return osp.isfile(_cont)


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z



def load_models(opt, models, checkpoints, device):
    out = 0
    ck = [osp.join(opt.output_dir, k) for k in checkpoints]
    for i, model in enumerate(models):
        if osp.isfile(ck[i]):
            model.load_state_dict(torch.load(ck[i], map_location=device))
            out += 1
    return out


def train(rank, world_size, opt):
    _start_time = time.time()
    torch.manual_seed(0)

    # logger
    log = PLog(osp.join(opt.output_dir, "train.csv"))
    _ini_epoch = 0
    if log.len:
        _ini_epoch = int(log.values["Epoch"] + 1)
        assert opt.continue_training == 1, f"{log.len} training steps recorded in {opt.output_dir}, set continue_training = 1, change or delete output_dir"
    
    _continue(opt.output_dir, init=True)

    # setup hw use
    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    # get options
    curriculum = getattr(curriculums, opt.curriculum)
    if opt.dataset_path != "":
        curriculum["dataset_path"]= opt.dataset_path
    if opt.batch_size > 0:
        curriculum["batch_size"] = opt.batch_size
    assert osp.isdir(osp.split(curriculum["dataset_path"])[0]), f"dataset path {curriculum['dataset_path']} not found"
    metadata = curriculums.extract_metadata(curriculum, 0)

    # setup fid evaluation
    if rank == 0:
        generated_dir = osp.join(opt.output_dir, 'evaluation/generated')
        fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, dataset_path=metadata["dataset_path"], target_size=128)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])
    SIREN = getattr(siren, metadata['model'])
    CHANNELS = 3

    # define models
    scaler = torch.cuda.amp.GradScaler()
    generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
    discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    # default: continue training
    # load checkpoints
    if opt.continue_training:
        _models = (generator, discriminator, ema, ema2)
        _checkpoints = ["generator.pth","discriminator.pth", 'ema.pth', 'ema2.pth']
        _loaded = load_models(opt, _models, _checkpoints, device)
        if log.len:
            assert _loaded == len(_models), f"Incomplete data, could not load all models {_checkpoints}"

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.continue_training:
        _models = [optimizer_G, optimizer_D]
        _checkpoints = ["optimizer_G.pth","optimizer_D.pth"]
        if not metadata.get('disable_scaler', False):
            _models.append(scaler)
            _checkpoints.append('scaler.pth')

        _loaded = load_models(opt, _models, _checkpoints, device)
        if log.len:
            assert _loaded == len(_models), f"Incomplete data, could not load all models {_checkpoints}"

    generator_losses = []
    discriminator_losses = []

    generator.step = log.len
    discriminator.step = log.len
    generator.epoch = _ini_epoch
    discriminator.epoch = _ini_epoch

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(osp.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None

    for epoch in range (_ini_epoch, opt.n_epochs):

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)
            if epoch == _ini_epoch:
                print("\ndataloader", len(dataloader), "batch_size", dataloader.batch_size)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

        for i, (imgs, _) in enumerate(dataloader):
            _time = time.time()
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                save_models(models=[ema, ema2, generator_ddp.module, discriminator_ddp.module,
                                    optimizer_G, optimizer_D, scaler],
                            names=['ema.pth', 'ema2.pth', 'generator.pth', 'discriminator.pth',
                                   'optimizer_G.pth', 'optimizer_D.pth', 'scaler.pth'],
                            folder=opt.output_dir,  prefix=True)
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z, **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs, gen_positions = generator_ddp(subset_z, **metadata)
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty = 0

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())


            if rank == 0:
                _loop = sround(time.time() - _time)
                _total_time = round((time.time() - _start_time))
                log.collect(Epoch=discriminator.epoch, Step=discriminator.step, D_Loss=sround(d_loss.item(),2), G_Loss=sround(g_loss.item(),2), Alpha=sround(alpha,2), ImgSz=metadata['img_size'])
                log.collect(Time=_loop, Total_Time=_total_time)
                log.write()

                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], osp.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], osp.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=5, normalize=True)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], osp.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], osp.join(opt.output_dir, f"{discriminator.step}_tilted_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            gen_imgs = generator_ddp.module.staged_forward(torch.randn_like(fixed_z).to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], osp.join(opt.output_dir, f"{discriminator.step}_random.png"), nrow=5, normalize=True)

                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    save_models(models=[ema, ema2, generator_ddp.module, discriminator_ddp.module, optimizer_G, optimizer_D, scaler],
                            names=['ema.pth', 'ema2.pth', 'generator.pth', 'discriminator.pth', 'optimizer_G.pth', 'optimizer_D.pth', 'scaler.pth'],
                            folder=opt.output_dir,  prefix=False)
                    torch.save(generator_losses, osp.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, osp.join(opt.output_dir, 'discriminator.losses'))

            # evaluate FID
            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp.module, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                    with open(osp.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

        if not _continue(opt.output_dir):
            break


    save_models(models=[ema, ema2, generator_ddp.module, discriminator_ddp.module, optimizer_G, optimizer_D, scaler],
                names=['ema.pth', 'ema2.pth', 'generator.pth', 'discriminator.pth', 'optimizer_G.pth', 'optimizer_D.pth', 'scaler.pth'],
                folder=opt.output_dir,  prefix=False)

    cleanup()

def save_models(models, names, folder, prefix=False):
    now = ""
    if prefix:
        now = datetime.now()
        now = now.strftime("%d--%H:%M--")
    prefix = now
    for i,  model in enumerate(models):
        torch.save(model.state_dict(), osp.join(folder, f"{now}{names[i]}"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--continue_training', type=int, default=1) # if 0, but output_dir exists... assert
    parser.add_argument('--batch_size', type=int, default=28) # ~28 images per 24GB ram

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(osp.join(opt.output_dir, 'evaluation/generated'), exist_ok=True)
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
