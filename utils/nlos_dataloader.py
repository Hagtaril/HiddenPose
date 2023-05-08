import os
import sys
sys.path.append('./')
# from lib.visualizer import joints_log
# from lib.vis_3view import vis_3view

import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from einops import rearrange


from config.config_noise import _C as cfg
    

class NlosDataset(Dataset):
    
    def __init__(self, cfg, datapath):
        super().__init__()
        # self.pan = cfg.DATASET.PAN
        # self.ratio = cfg.DATASET.RATIO
        self.vol_size = cfg.DATASET.VOL_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.downsample_cnt = cfg.DATASET.DAWNSAMPLE_CNT

        self.measFiles = []
        self.volFiles = []
        self.jointsFils = []
        self.volPath = os.path.join(datapath, 'vol')
        self.jointsPath = os.path.join(datapath, 'joints')
        self.measPath = os.path.join(datapath, 'meas')
        measNames = os.listdir(self.measPath)
        # volNames = os.listdir(self.volPath)
        for measName in measNames:
            assert os.path.splitext(measName)[1] == '.hdr', \
                f'Data type should be .hdr,not {measName} in {self.measPath}'
            measFile = os.path.join(self.measPath, measName)
            self.measFiles.append(measFile)

            volFile = os.path.join(self.volPath, os.path.splitext(measName)[0] + '.npy')
            assert os.path.isfile(volFile), \
                f'Do not have related vol {volFile}'
            self.volFiles.append(volFile)
            jointsFile = os.path.join(self.jointsPath, os.path.splitext(measName)[0] + '.joints')
            assert os.path.isfile(jointsFile), \
                f'Do not have related joints {jointsFile}'
            self.jointsFils.append(jointsFile)

    def __getitem__(self, index):
        measFile = self.measFiles[index]
        volFile = self.volFiles[index]
        jointFile = self.jointsFils[index]

        try:
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
        except TypeError:
            measFile = self.measFiles[0]
            jointFile = self.jointsFils[0]

            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is TypeError. \n--------------------------\n')
        except:
            measFile = self.measFiles[0]
            jointFile = self.jointsFils[0]

            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is wrong. \n--------------------------\n')

        meas = rearrange(meas, '(t h) w ->t h w', t=600)[:512]
        vol = np.load(volFile).astype(np.float32)
        joints = np.loadtxt(jointFile)

        # meas = zoom(meas[:512, :, :], [0.5, 1, 1])  # too slow
        meas = (meas[::2] + meas[1::2]) / 2  # [256,256,256]
        for i in range(self.downsample_cnt):
            meas = (meas[::2] + meas[1::2]) / 2
            meas = (meas[:,::2] + meas[:,1::2]) / 2
            meas = (meas[:,:,::2] + meas[:,:,1::2]) / 2

            vol = (vol[::2] + vol[1::2]) / 2
            vol = (vol[:,::2] + vol[:,1::2]) / 2
            vol = (vol[:,:,::2] + vol[:,:,1::2]) / 2

        
        # joints[:,0] = (joints[:,0] + self.pan) * self.vol_size[0] / self.ratio
        # joints[:,1] = (joints[:,1] + self.pan) * self.vol_size[1] / self.ratio
        # joints[:,2] = (joints[:,2] + self.pan) * self.vol_size[2] / self.ratio
        # joints[:, 0] = (joints[:, 0]*165+128)
        # joints[:, 1] = 256-(joints[:, 1]*165+128)
        # joints[:, 2] = 223-(joints[:, 2]*165+128)
        joints[:, 0] = (joints[:, 0]*128+128)
        joints[:, 1] = 256-(joints[:, 1]*128+128)
        joints[:, 2] = 225-(joints[:, 2]*128+128)
        w = joints[:, 0].copy()
        h = joints[:, 1].copy()
        d = joints[:, 2].copy()
        joints[:, 0] = d
        joints[:, 1] = h
        joints[:, 2] = w
        _, person_name = os.path.split(measFile)
        person_id, _ = os.path.splitext(person_name)
        return rearrange(meas, 't h w -> 1 t h w'), rearrange(vol, 'd h w -> 1 d h w'),  \
               joints / (self.vol_size[0] / self.heatmap_size[0]), person_id


    def __len__(self):
        
        return len(self.volFiles)



if __name__ == '__main__':
    # a = np.random.random((224, 224))
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # print(a.sum())

    # plt.show()
    testDataLoader = True
    if testDataLoader:
        datapath = "/data1/nlospose/zip/train/"
        train_data = NlosDataset(cfg, datapath)
        trainloader = DataLoader(train_data, batch_size=1, shuffle=True)

        print(type(trainloader))
        dataiter = iter(trainloader)
        meas, vol, joints, person_id = dataiter.next()

        im = np.zeros((256,256))
        for i in range(24):
            x = np.int64(joints[0,i,0])
            y = np.int64(joints[0,i,1])
            im[x:x+3,y:y+3] = 10
        plt.figure()
        plt.imshow(np.rot90(im))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig('./joints.jpg')
        
        plt.figure()
        im2 = vol[0,0].sum(1)
        plt.imshow(np.rot90(im2))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig('./vol_proj.jpg')
        
        plt.figure()
        plt.plot(meas[0,:,100,100])
        plt.savefig('./meas.jpg')
        print(f'meas\' type is {type(vol)}:\n{vol.shape}\n----------------')
        # print(joints.shape)
        # vis_3view(meas)
        from feature_propagation import FeaturePropagation

        model = FeaturePropagation(
			time_size=cfg.MODEL.TIME_SIZE // cfg.MODEL.TIME_DOWNSAMPLE_RATIO,
			image_size=cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.IMAGE_DOWNSAMPLE_RATIO,
			wall_size=cfg.MODEL.WALL_SIZE,
			bin_len=cfg.MODEL.BIN_LEN * cfg.MODEL.TIME_DOWNSAMPLE_RATIO,
			dnum=cfg.MODEL.DNUM,
			dev=cfg.DEVICE,
		).to('cuda:2')

        output = model(meas.to('cuda:2'))
        
        from vis_3view import vis
        
        print("finish")

    # testGetHeatmap = False
    # heatmap = GetHeatmap()
    # if testGetHeatmap:
    #
    #     joints = np.loadtxt(r"D:\LPP\nlospose\data\nlospose\train\joints\light-1-411.9427-4.0100-100.0000.joints")
    #     joints_vis = np.ones_like(joints)
    #
    #     a, b = heatmap.generate_target(joints, joints_vis)
    #     print(a[0].sum())
    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(a[0] * 1000, cmap='hot')
    #     plt.show()
    #     for (x, y), item in np.ndenumerate(a[1]):
    #         if item > 0:
    #             print(f'({x}, {y}) : {item}')
    #     print(a.shape)
    #
    # testGetHeatmap3D = True
    # # if testGetHeatmap3D:
    # #     heatmap.generate_3dtarget()
    

