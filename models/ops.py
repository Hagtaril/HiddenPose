

import torch.nn.functional as F
import torch.nn as nn


class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re



#####################################################################
# if __name__ == '__main__':
#
#     import numpy as np
#
#     btf = torch.from_numpy(np.random.rand(10, 16, 256, 256).astype(np.float32)).cuda()
#
#     scaledownlayer = Interpsacle2d(0.5)
#     ctf = scaledownlayer(btf)
#
#     scaleuplayer = Interpsacle2d(2.0)
#     dtf = scaleuplayer(ctf)
#
#     print(dtf.shape)
    
