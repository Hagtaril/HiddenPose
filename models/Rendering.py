import torch
from torch import nn

import numpy as np

class Rendering(nn.Module):

    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()

        ######################################
        assert out_channels == 1

        weights = np.zeros((1, 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, 1:, :, :] = 1.0
        else:
            weights[:, :1, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)

        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)

        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + 1,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )

    def forward(self, x0):

        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:1, :, :]
        x0_dep = x0[:, dim:dim + 1, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)

        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)

        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)

        re = x0_conv_256 + debugvalue * x2

        return re


if __name__ == '__main__':
    layer = Rendering(nf0=2, out_channels=out_channels // 2)