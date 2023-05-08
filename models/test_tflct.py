import torch

from tflct import *
# from tflct import lct
import cv2
import numpy as np
from scipy import ndimage
from einops import rearrange, repeat
from utils.PlotVolume import PlotVolume

imname = r'D:\LPP\cuda-render\data\bunny-renders\0\bunny\shine_0.0000-rot_0.0000_0.0000_0.0000-shift_0.0000_0.0000_0.0000\light-1-411.9427-4.0100-100.0000.hdr'

# parameters
gray = True
imszs = np.array([600,256,256,1])
time_cropbe = 0
time_cropen = 6
clip = False
videoconfocal = 1
clipratio = 0.97

im = cv2.imread(imname, -1)
im = im / np.max(im)

if gray:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / np.max(im)

# it should be txhxwx3
im = im.reshape(*imszs)

##################################################################
# crop meaningful region
timebe = int(100 * time_cropbe)
timeen = int(100 * time_cropen)
imcrop = im[timebe:timeen, :, :, :]

if clip:
    # do we need to do the fixed normalization?
    data90 = np.percentile(imcrop, clipratio)
    maxdata = np.max(imcrop)
    meandata = np.mean(imcrop)
    # data90 = maxdata
    imnorm = imcrop / data90
    imnorm[imnorm > 1] = 1
else:
    imnorm = imcrop

# smooth to get rid of the artifacts
a = ndimage.filters.uniform_filter
a = ndimage.gaussian_filter

if False and (videoconfocal <= 1):
    sig = 0.7
    if gray:
        imsmooth = a(imnorm[:, :, :, 0], sigma=sig, mode='reflect', cval=0.0)
    else:
        imb = a(imnorm[:, :, :, 0], sigma=sig, mode='reflect', cval=0.0)
        img = a(imnorm[:, :, :, 1], sigma=sig, mode='reflect', cval=0.0)
        imr = a(imnorm[:, :, :, 2], sigma=sig, mode='reflect', cval=0.0)
        imsmooth = np.stack([imb, img, imr], axis=3)
else:
    if gray:
        imsmooth = imnorm[:, :, :, 0]
    else:
        imsmooth = imnorm

crop = 512
bin_len = 0.01
imsmooth =imsmooth[:512,:,:]

imsmooth = rearrange(imsmooth, 't w h -> w h t')
K = 1
for k in range(K):
    # imsmooth = imsmooth[::2, :, :] + imsmooth[1::2, :, :]
    # imsmooth = imsmooth[:, ::2, :] + imsmooth[:, 1::2, :]

    imsmooth = imsmooth[:, :, ::2] + imsmooth[:, :, 1::2]
    crop = crop // 2
    bin_len = bin_len * 2

input = rearrange(imsmooth, 'h w t -> 1 1 t h w')
input = torch.from_numpy(input).cuda()
lct_layer = lct(spatial=input.shape[-1], crop=input.shape[2], bin_len=bin_len, wall_size=2.0, method='lct')
dev = 'cuda'
dnum = 1
lct_layer.todev(dev, dnum)

tbe = 0
tlen = crop
for i in range(10):
    print(i)
    re = lct_layer(input, [tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen])

volumn_MxNxN = re.detach().cpu().numpy()[0, -1]

# get rid of bad points
zdim = volumn_MxNxN.shape[0] * 100 // 128
volumn_MxNxN = volumn_MxNxN[:zdim]
print('volumn min, %f' % volumn_MxNxN.min())
print('volumn max, %f' % volumn_MxNxN.max())
# volumn_MxNxN[:5] = 0
# volumn_MxNxN[-5:] = 0

volumn_MxNxN[volumn_MxNxN < 0] = 0

PlotVolume(volumn_MxNxN, 0.1)

front_view = np.max(volumn_MxNxN, axis=0)
cv2.imshow("front", front_view / np.max(front_view))
# cv2.imshow("gt", imgt)
cv2.waitKey()

left_view = np.max(volumn_MxNxN, axis=1)
cv2.imshow("left", left_view / np.max(left_view))
cv2.waitKey()

top_view = np.max(volumn_MxNxN, axis=2)
cv2.imshow("top", top_view / np.max(top_view))
cv2.waitKey()