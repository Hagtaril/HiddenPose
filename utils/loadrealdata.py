import torch
from scipy.io import loadmat
from einops import rearrange


def load_realdata(datapath = "data/lct256_human.mat", downsample_cnt = 1):
    input = loadmat(datapath)
    meas = input['data_new']
    meas = rearrange(meas, 'h w t -> t w h')
    meas = (meas[::2] + meas[1::2]) / 2  # [256,256,256]
    for i in range(downsample_cnt):
        meas = (meas[::2] + meas[1::2]) / 2
        meas = (meas[:,::2] + meas[:,1::2]) / 2
        meas = (meas[:,:,::2] + meas[:,:,1::2]) / 2
    return torch.tensor(meas)


if __name__=="__main__":

    data = load_realdata()

    print("finished")