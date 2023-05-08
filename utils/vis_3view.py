import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def vis_3view(re, name):
    volumn_MxNxN = re.detach().cpu().numpy()[0,-1]

    # get rid of bad points
    # if name == 'output':
    #     zdim = volumn_MxNxN.shape[0]
    # elif name == 'volume':
    zdim = volumn_MxNxN.shape[0]
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0

    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    plt.imshow(front_view / np.max(front_view))
    path = Path('./plot')
    Path.mkdir(path, exist_ok=True)
    plt.savefig(f'./plot/{name}_front_view.jpg')

    top_view = np.max(volumn_MxNxN, axis=1)
    plt.imshow(np.rot90(top_view / np.max(top_view), 2))
    plt.savefig(f'./plot/{name}_top_view.jpg')

    left_view = np.max(volumn_MxNxN, axis=2)
    plt.imshow(np.rot90(left_view / np.max(left_view), 3))
    plt.savefig(f'./plot/{name}_left_view.jpg')


def vis_3view_cv(re):
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]

    # get rid of bad points
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0

    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    cv2.imshow("front", front_view / np.max(front_view))
    # cv2.imshow("gt", imgt)
    cv2.waitKey()

    left_view = np.max(volumn_MxNxN, axis=1)
    cv2.imshow("left", left_view / np.max(left_view))
    cv2.waitKey()

    left_view = np.max(volumn_MxNxN, axis=2)
    cv2.imshow("top", left_view / np.max(left_view))
    cv2.waitKey()


if __name__ == '__main__':
    vol = torch.ones((128,256,256)).to('cuda')
    vis_3view(vol)
    print("finish")