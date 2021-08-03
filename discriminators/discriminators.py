"""Discrimators used in pi-GAN"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sgdiscriminators import *

# pylint: disable=no-member

class ResidualCoordConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample

    def forward(self, identity):
        y = self.network(identity)

        if self.downsample: y = nn.functional.avg_pool2d(y, 2)
        if self.downsample: identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)

        y = (y + identity)/math.sqrt(2)
        return y


class ProgressiveDiscriminator(nn.Module):
    """Implement of a progressive growing discriminator with ResidualCoordConv Blocks"""

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCoordConvBlock(16, 32, downsample=True),   # 512x512 -> 256x256
            ResidualCoordConvBlock(32, 64, downsample=True),   # 256x256 -> 128x128
            ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
            ResidualCoordConvBlock(128, 256, downsample=True), # 64x64   -> 32x32
            ResidualCoordConvBlock(256, 400, downsample=True), # 32x32   -> 16x16
            ResidualCoordConvBlock(400, 400, downsample=True), # 16x16   -> 8x8
            ResidualCoordConvBlock(400, 400, downsample=True), # 8x8     -> 4x4
            ResidualCoordConvBlock(400, 400, downsample=True), # 4x4     -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(16),
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}


    def forward(self, input, alpha, instance_noise=0, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]

        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x

class ProgressiveEncoderDiscriminator(nn.Module):
    """
    Implement of a progressive growing discriminator with ResidualCoordConv Blocks.
    Identical to ProgressiveDiscriminator except it also predicts camera angles and latent codes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCoordConvBlock(16, 32, downsample=True),   # 512x512 -> 256x256
            ResidualCoordConvBlock(32, 64, downsample=True),   # 256x256 -> 128x128
            ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
            ResidualCoordConvBlock(128, 256, downsample=True), # 64x64   -> 32x32
            ResidualCoordConvBlock(256, 400, downsample=True), # 32x32   -> 16x16
            ResidualCoordConvBlock(400, 400, downsample=True), # 16x16   -> 8x8
            ResidualCoordConvBlock(400, 400, downsample=True), # 8x8     -> 4x4
            ResidualCoordConvBlock(400, 400, downsample=True), # 4x4     -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(16),
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2)
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}


    def forward(self, input, alpha, instance_noise=0, **kwargs):
        if instance_noise > 0:
            input = input + torch.randn_like(input) * instance_noise

        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        latent = x[..., 1:257]
        position = x[..., 257:259]

        return prediction, latent, position
