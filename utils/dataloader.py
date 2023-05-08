import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import zoom
import math
import os
import cv2
from einops import rearrange
import matplotlib.pyplot as plt
from scipy import signal
import sys
from scipy.ndimage import gaussian_filter



# TODO preprocess
class GetHeatmap():
    def __init__(self):
        self.num_joints = 24
        self.target_type = 'gaussian'
        self.image_size = np.array([1,1,1])
        self.heatmap_size = np.array([224, 224, 224])
        self.sigma = 1
        self.use_different_joints_weight = False

    def generate_target_3d(self, joints, joints_vis):
        # '''
        # :param joints:  [num_joints, 3]
        # :param joints_vis: [num_joints, 3]
        # :return: target, target_weight(1: visible, 0: invisible)
        # '''
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


    def generate_target(self, joints, joints_vis):
        # '''
        # :param joints:  [num_joints, 3]
        # :param joints_vis: [num_joints, 3]
        # :return: target, target_weight(1: visible, 0: invisible)
        # '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight



class NlosPoseDataset(Dataset):
    """docstring for NlosPoseDataset"""

    def __init__(self,cfg, datapath):
        super().__init__()
        self.sigma = cfg.DATASET.SIGMA
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        self.simdr_split_ratio = cfg.DATASET.SIMDR_SPLIT_RATIO
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.use_different_joints_weight = True
        self.sigma = cfg.DATASET.SIGMA
        self.joints_weight  = 1

        self.measFiles = []
        self.jointsFiles = []
        self.measPath = os.path.join(datapath, 'meas')
        self.jointsPath = os.path.join(datapath, 'joints')
        measNames = os.listdir(self.measPath)
        for measName in measNames:
            assert os.path.splitext(measName)[1] == '.hdr', \
                f'Data type should be .hdr,not {measName} in {self.measPath}'
            fileName = os.path.join(self.measPath, measName)
            self.measFiles.append(fileName)
            jointsName = os.path.join(self.jointsPath, os.path.splitext(measName)[0] + '.joints')
            assert os.path.isfile(jointsName), \
                f'Do not have related joints {jointsName}'
            self.jointsFiles.append(jointsName)
        # meas = cv2.imread(fileName, -1)
        # self.meas = meas / np.max(meas)


    def __getitem__(self, index):
        # print(index)
        measFile = self.measFiles[index]
        jointFile = self.jointsFiles[index]
        try:
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
        except TypeError:
            measFile = self.measFiles[0]
            jointFile = self.jointsFiles[0]

            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(f'--------------------\nNo.{index} meas is wrong. \n--------------------------\n')
        except :
            measFile = self.measFiles[0]
            jointFile = self.jointsFiles[0]

            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(f'--------------------\nNo.{index} meas is wrong. \n--------------------------\n')

        joints = np.loadtxt(jointFile)
        # coord to box
        meas = rearrange(meas, '(t h) w ->t h w', t = 600)
        meas = zoom(meas[:512], [0.5, 1, 1])
        # meas = meas[:512]
        # joints = np.int64((joints + 0.5) * 256)
        joints[:,0] = np.int64((joints[:,0] + 0.5) * 64)
        joints[:,1] = np.int64((joints[:,1] + 0.5) * 64)
        joints[:,2] = np.int64((joints[:,2] + 0.5) * 64)
        # x, y, z, w = self.generate_sa_simdr(joints)
        meas = rearrange(meas, 't h w-> 1 t h w')

        # return torch.Tensor(meas), torch.Tensor(joints), torch.Tensor(x), torch.Tensor(y), torch.Tensor(z), torch.Tensor(w)
        return torch.Tensor(meas), torch.Tensor(joints.reshape(joints.shape[0], -1)), measFile

    def __len__(self):
        return len(self.measFiles)

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = image_size / heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        mu_z = joint[2]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z - tmp_size + 1)]
        if ul[0] >= (self.image_size[0] * self.simdr_split_ratio) or ul[1] >= self.image_size[1] * self.simdr_split_ratio or ul[2] >= self.image_size[2] * self.simdr_split_ratio \
                or br[0] < 0 or br[1] < 0 or br[2] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight

    def generate_sa_simdr(self, joints, joints_vis = None):
        '''
        :param joints:  [self.num_joints, 3]
        :param joints_vis: [self.num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        if torch.is_tensor(joints):
            joints = joints.numpy()
        if torch.is_tensor(joints_vis):
            joints_vis = joints_vis.numpy()


        if joints_vis is None:
            joints_vis = np.ones((self.num_joints, 3))
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target_x = np.zeros((self.num_joints,
                             int(self.image_size[0] * self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.num_joints,
                             int(self.image_size[1] * self.simdr_split_ratio)),
                            dtype=np.float32)
        target_z = np.zeros((self.num_joints,
                             int(self.image_size[2] * self.simdr_split_ratio)),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0] * self.simdr_split_ratio
            mu_y = joints[joint_id][1] * self.simdr_split_ratio
            mu_z = joints[joint_id][2] * self.simdr_split_ratio

            x = np.arange(0, int(self.image_size[0] * self.simdr_split_ratio), 1, np.float32)
            y = np.arange(0, int(self.image_size[1] * self.simdr_split_ratio), 1, np.float32)
            z = np.arange(0, int(self.image_size[2] * self.simdr_split_ratio), 1, np.float32)

            v = target_weight[joint_id]
            if v > 0.5:
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2))) / (
                        self.sigma * np.sqrt(np.pi * 2))
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2))) / (
                        self.sigma * np.sqrt(np.pi * 2))
                target_z[joint_id] = (np.exp(- ((z - mu_z) ** 2) / (2 * self.sigma ** 2))) / (
                        self.sigma * np.sqrt(np.pi * 2))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_x, target_y, target_z, target_weight



if __name__ == '__main__':
    # a = np.random.random((224, 224))
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # print(a.sum())

    # plt.show()
    testDataLoader = True
    if testDataLoader:
        datapath = r'D:\LPP\nlospose_test\data\nlospose\train'
        import sys
        print(sys.path)
        sys.path.append(r'D:\LPP\nlospose_test\models')
        from config import _C as cfg
        train_data = NlosPoseDataset(cfg, datapath)
        trainloader = DataLoader(train_data, batch_size=2, shuffle=True)

        print(type(trainloader))
        dataiter = iter(trainloader)
        meas, x, y, z, w = dataiter.next()
        print(f'meas\' type is {type(meas)}:\n{meas.shape}\n----------------')
        # print(joints.shape)


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

