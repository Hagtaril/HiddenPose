import torch
import torch.nn as nn
import numpy as np
import  cv2
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg
from models.customer_layers_3 import Transient2volumn

from models.tfmodule import diffmodule as lct






###########################################################################
def normalize(data_bxcxdxhxw):
    b, c, d, h, w = data_bxcxdxhxw.shape
    data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
    
    data_min = data_bxcxk.min(2, keepdim=True)[0]
    data_zmean = data_bxcxk - data_min
    
    # most are 0
    data_max = data_zmean.max(2, keepdim=True)[0]
    data_norm = data_zmean / (data_max + 1e-15)
    
    return data_norm.view(b, c, d, h, w)


################################################################

class NlosFeature(nn.module):
    def __init__(self, cfg):
        super(NlosFeature, self).__init__()
        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.out_channels = cfg.MODEL.OUT_CHANNELS
        self.grid_dim = cfg.MODEL.GRID_DIM
        self.img_side_length = cfg.MODEL.IMAGE_SIZE[0]
        self.mode = cfg.MODEL.MODE 

        self.bin_len = cfg.MODEL.CONST.BIN_LEN
        self.wall_size = cfg.MODEL.CONST.WALL_SIZE
        self.crop = cfg.MODEL.CROP

####################################################################
class DeepVoxels(nn.Module):

    def __init__(self,
                 nf0=16,
                 in_channels=3,
                 out_channels=3,
                 img_sidelength=256,
                 crop = 256,
                 grid_dim=32,
                 bin_len=0.01,
                 wall_size=2.0,
                 mode='fk',
                 res0=0):
        
        super(DeepVoxels, self).__init__()
        
        ###################################33
        # 4 networks
        # 1 downsample
        # 2 unet
        # 3 occlusion
        # 4 render
        
        self.crop = crop
        imsz = 256
        assert imsz == img_sidelength
        
        volumnsz = 128
        assert volumnsz == grid_dim
        sres = imsz // volumnsz
        
        tfull = 512
        tsz = 128
        volumntsz = 64
        tres = tsz // volumntsz
        # assert sres == tres
        
        ########################################################
        basedim = nf0
        self.basedim = basedim
        # assert not raytracing
        
        self.downnet = Transient2volumn(nf0=basedim, in_channels=in_channels)
        
        print('bin_len %.7f' % bin_len)
        self.lct = lct(spatial=imsz // sres, crop=self.crop ,
                       bin_len=bin_len * tres, \
                       mode=mode, wall_size=wall_size)
        
        layernum = 0
        # self.visnet = VisibleNet(nf0=basedim * 1 + 1, layernum=layernum)
        
        # self.depth = True
        assert out_channels == 6 or out_channels == 2
        # self.rendernet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=out_channels // 2)
        # self.depnet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=out_channels // 2, isdep=True)
    
    def todev(self, dev):
        self.lct.todev(dev, self.basedim * 1 + 1)
                
    def noise(self, data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

    def forward(self, input_voxel, tbes, tens):
        
        if False:
            noisedata = self.noise(input_voxel)
        else:
            noisedata = input_voxel
        
        ###############################
     # TO BE DO : find tbes & tens

        data_norm = normalize(noisedata)
        
        tfre = self.downnet(data_norm)
        
        # lct
        tfre2 = self.lct(tfre, tbes, tens)

        x = tfre2
        zdim = x.shape[2]
        zdimnew = zdim * 100 // 128
        x = x[:, :, :zdimnew]
        tfre2 = x
        
        tfre2 = nn.ReLU()(tfre2)
        tfre2 = normalize(tfre2)
        tfre2 = tfre2 * 1.e5

        return tfre2


#################################################################
if __name__ == '__main__':
    
    basedim = 1
    tres = 2
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    frame = 512
    in_channels = 1
    data = np.zeros((1, in_channels, frame, 256, 256), dtype=np.float32)
    
    # from scipy.io import loadmat
    # data = loadmat(file_name='/home/wenzheng/largestore/nlos-phasor/realdata/resolution0.mat')
    # rect_data_hxwxt = data['measlr']
    rect_data_hxwxt = np.random.randn(256, 256, 512)
    rect_data_txhxw = np.transpose(rect_data_hxwxt, axes=[2, 0, 1])
    data = rect_data_txhxw.reshape(1, 1, 512, 256, 256)
    tfdata = torch.from_numpy(data).to(dev).float()
   
    model = DeepVoxels(
                       nf0=basedim,
                       in_channels=in_channels,
                       out_channels=2,
                       img_sidelength=256,
                       grid_dim=128,
                       mode='lct')
    model = model.to(dev)
    model.todev(dev)
    re = model(tfdata, [0, 0, 0, 0, 0], [frame // tres, 32, 32, 32, 32])
    print('\n')
    print(re.shape)
    print('\n')
    
    re = re.detach().cpu().numpy()
    re = (re + 1) / 2
    im = re[0, 0]
    dep = re[0, 1]
    im = im / np.max(im)
    
    import cv2
    cv2.imshow('im', im)
    cv2.imshow('dep', dep)
    cv2.waitKey()

    
    
