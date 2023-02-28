import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tflct import lct
import sys

# sys.path.append('../utils_pytorch')
# from tfmodule import diffmodule as lct

from models.ops import ResConv3D
import numpy as np

debugvalue = 1


#################################################################################
class Transient2volumn(nn.Module):
    
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn, self).__init__()
        
        ###############################################
        assert in_channels == 1
        
        weights = np.zeros((1, 1, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    
    def forward(self, x0):
        
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        
        x1 = self.conv1(x0)
        
        re = torch.cat([x0_conv, x1], dim=1)
        
        return re


############################################################
if __name__ == '__main__':
    
    tfull = 512
    
    imsz = 256
    tsz = 128
    volumnsz = 128
    volumntsz = 64
    
    sres = imsz // volumnsz
    tres = tsz // volumntsz
    
    basedim = 1
    bnum = 1
    channel = 1
    
    ####################################################
    dev = 'cuda:0'
    data = np.zeros((bnum, channel, tsz, imsz, imsz), dtype=np.float32)
    
    downnet = Transient2volumn(nf0=basedim, in_channels=channel)
    downnet = downnet.to(dev)
    tfdata = torch.from_numpy(data).to(dev)
    tfre = downnet(tfdata)
    # tfre = nn.ConstantPad3d((0, 0, 0, 0, 2, 3), 0)(tfre)
    print('\n')
    print(tfre.shape)
    print('\n')
    
    # next unet
    unet3d = lct(spatial=imsz // sres, crop=tfull // tres, bin_len=0.01 * tres, \
                 mode='lct')
    unet3d = unet3d.to(dev)
    unet3d.todev(dev, basedim * 1 + 1)
    
    tbes = [0] * bnum
    tens = [tsz // tres] * bnum
    tfre2 = unet3d(tfre, tbes, tens)
    print('\n')
    print(tfre2.shape)
    print('\n')
    
    layernum = 0
    downnet2 = VisibleNet(nf0=basedim * 1 + 1, layernum=layernum)
    downnet2 = downnet2.to(dev)
    tfre2 = downnet2(tfre2)
    print('\n')
    print(tfre2.shape)
    print('\n')
    
    rendernet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=1)
    rendernet = rendernet.to(dev)
    
    rendered_img = rendernet(tfre2)
    print('\n')
    print(rendered_img.shape)
    print('\n')
    
