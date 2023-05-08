import numpy as np 
import scipy.sparse as ssp
import torch
import torch.nn as nn 
import torch.nn.functional as F 

'''
self.lct = lct(
	spatial = imsz // sres,
	crop = self.time_size,
	bin_len=bin_len * tres,
	mode = mode,
	wall_size = wall_size
)
'''


class FeaturePropagation(nn.Module):
	"""docstring for FeaturePropagation
	Input : Tensor (batch_size, channels, time_size, image_size[0], image_size[1])
	----------
	Output : Tensor (batch_size, channels, time_size, image_size[0], image_size[1])
	"""

	def __init__(
			self,
			image_size=256,
			time_size=512,
			bin_len=0.01,
			wall_size=2.0,
			mode='lct',
			material='diffuse',
			dnum = 1,
			dev = 'cpu',
	):
		super(FeaturePropagation, self).__init__()
		assert mode == 'lct', \
			f'{mode} is not spported. Feature propagation only support lct by now'

		self.method = LCT(int(image_size), time_size, bin_len, wall_size, mode=mode, material=material)
		self.method.todev(dev, dnum)

	def forward(self, x, time_begin, time_end):
		return self.method(x, time_begin, time_end)

class LCT(nn.Module):
	"""docstring for LCT"""
	def __init__(
		self,
		image_size = 256,
		time_size = 128,
		bin_len = 0.01,
		wall_size = 2.0,
		mode  = 'lct',
		material = 'diffuse'
	):
		super(LCT, self).__init__()
		
		self.image_size = image_size
		self.time_size = time_size
		assert 2 ** int(np.log2(self.time_size)) == time_size, \
			f'time size should be a power of 2'
		self.bin_len = bin_len
		self.wall_size = wall_size

		self.mode = mode
		self.material = material

		self._parpareparam()

	def _parpareparam(self):
		self.c = 3e8
		self.width = self.wall_size / 2.
		self.bin_resolution = self.bin_len / self.c 
		self.trange = self.time_size * self.c * self.bin_resolution

		self.snr = 1e-1

		temprol_grid = self.time_size
		sptial_grid = self.image_size

		gridz_M = np.arange(temprol_grid, dtype=np.float32)
		gridz_M = gridz_M / (temprol_grid - 1)
		gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
		self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))

		slope = self.width / self.trange
		psf = self._definePsf(sptial_grid, temprol_grid, slope)
		fpsf = np.fft.fftn(psf)

		if self.mode == 'lct':
			invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
		elif self.mode == 'bp':
			invpsf = np.conjugate(fpsf)

		self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
		self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)

		mtx_MxM, mtxi_MxM = self._resamplingOperator(temprol_grid)
		self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
		self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))

		if self.mode == 'bp':
			lapw_kxkxk = filterLaplacian()
			k = lapw_kxkxk.shape[0]
			self.pad = nn.ReplicationPad3d(2)
			self.lapw = torch.from_numpy(lapw_kxkxk).reshape(1, 1, k, k, k)

		self.todev('cuda', 1)

	def _resamplingOperator(self, temprol_grid):

		M = temprol_grid
		row = M ** 2 
		col = M
		assert 2 ** int(np.log2(M)) == M

		# 1 to M^2
		x = np.arange(row, dtype=np.float32)
		x = x + 1

		rowidx = np.arange(row)
		# 0 to M-1
		colidx = np.ceil(np.sqrt(x)) - 1
		data = np.ones_like(rowidx, dtype=np.float32)
		row = int(row)
		col = int(col)
		mtx1 = ssp.csr_matrix((data, (rowidx, colidx)), shape=(row, col), dtype=np.float32)
		mtx2 = ssp.spdiags(data=[1.0 / np.sqrt(x)], diags=[0], m=row, n=row)

		mtx = mtx2.dot(mtx1)
		K = int(np.log2(M))
		for _ in np.arange(K):
			mtx = 0.5 * (mtx[0::2, :] + mtx[1::2])
		# mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

		mtxi = np.transpose(mtx)

		return mtx.toarray(), mtxi.toarray()

	def _definePsf(self, sptial_grid, temprol_grid, slope):
		# slop is time_range / wall_size
		N = sptial_grid
		M = temprol_grid

		x_2N = np.arange(2 * sptial_grid, dtype=np.float32)
		x_2N = x_2N / (2 * sptial_grid - 1) * 2 - 1

		y_2N = x_2N

		z_2M = np.arange(2 * temprol_grid, dtype=np.float32)
		z_2M = z_2M / (2 * temprol_grid - 1) * 2

		[gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(x_2N, y_2N, z_2M)

		a_2Nx2NX2M = (4 * slope) ** 2 * (gridx_2Nx2Nx2M ** 2 + gridy_2Nx2Nx2M ** 2) - gridz_2Nx2Nx2M
		b_2Nx2NX2M = np.abs(a_2Nx2NX2M)

		c_2Nx2NX2M = np.min(b_2Nx2NX2M, axis=2, keepdims=True)

		d_2Nx2NX2M = np.abs(b_2Nx2NX2M - c_2Nx2NX2M) < 1e-8
		d_2Nx2NX2M = d_2Nx2NX2M.astype(np.float32)

		e_2Nx2NX2M = d_2Nx2NX2M / np.sqrt(np.sum(d_2Nx2NX2M))
		# shift
		f1_2Nx2NX2M = np.roll(e_2Nx2NX2M, shift=N, axis=0)
		f2_2Nx2NX2M = np.roll(f1_2Nx2NX2M, shift=N, axis=1)

		psf_2Mx2Nx2N = np.transpose(f2_2Nx2NX2M, [2, 0, 1])

		return psf_2Mx2Nx2N

	def todev(self, dev, dnum):
		self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
		self.datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.time_size, 2 * self.image_size, 2 * self.image_size), dtype=torch.float32, device=dev)

		self.mtx_MxM_todev = self.mtx_MxM.to(dev)
		self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)

		self.invpsf_real_todev = self.invpsf_real.to(dev)
		self.invpsf_imag_todev = self.invpsf_imag.to(dev)

		if self.mode == 'bp':
			self.lapw_todev = self.lapw.to(dev)

	def forward(self, feture_bxdxtxhxw, tbes, tens):

		# 1 padd data with zero
		bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
		for tbe, ten in zip(tbes, tens):
			assert tbe >= 0
			assert ten <= self.time_size
		dev = feture_bxdxtxhxw.device

		featpad_bxdxtxhxw = []
		for i in range(bnum):
			featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
			featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.time_size - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
			featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
			featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
		featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)

		assert hnum == wnum
		assert hnum == self.image_size
		sptial_grid = hnum
		temprol_grid = self.time_size

		####################################################
		# 3 run lct
		# assert bnum == 1
		data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.time_size, hnum, wnum)

		gridz_1xMx1x1 = self.gridz_1xMx1x1_todev
		if self.material == 'diffuse':
			data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 4)
		elif self.material == 'specular':
			data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 2)

		datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
		datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)

		left = self.mtx_MxM_todev
		right = data_BDxTxHxW.view(bnum * dnum, temprol_grid, -1)
		tmp = torch.matmul(left, right)
		tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)

		datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
		datafre = torch.rfft(datapad_BDx2Tx2Hx2W, 3, onesided=False)
		datafre_real = datafre[:, :, :, :, 0]
		datafre_imag = datafre[:, :, :, :, 1]

		re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
		re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
		refre = torch.stack([re_real, re_imag], dim=4)
		re = torch.ifft(refre, 3)

		volumn_BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid, 0]

		left = self.mtxi_MxM_todev
		right = volumn_BDxTxHxW.reshape(bnum * dnum, temprol_grid, -1)
		tmp = torch.matmul(left, right)
		tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)

		volumn_BDxTxHxW = tmp2

		if self.mode == 'bp':
			volumn_BDx1xTxHxW = volumn_BDxTxHxW.unsqueeze(1)
			lapw = self.lapw_todev
			volumn_BDx1xTxHxW = self.pad(volumn_BDx1xTxHxW)
			volumn_BDx1xTxHxW = F.conv3d(volumn_BDx1xTxHxW, lapw)
			volumn_BDxTxHxW = volumn_BDx1xTxHxW.squeeze(1)
			if True:
				volumn_BDxTxHxW[:, :1] = 0

		volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.time_size, hnum, wnum)

		return volumn_BxDxTxHxW


def normalize(data_bxcxdxhxw):
	b, c, d, h, w = data_bxcxdxhxw.shape
	data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)

	data_min = data_bxcxk.min(2, keepdim = True)[0]
	data_zmean = data_bxcxk - data_min

	data_max = data_zmean.max(2, keepdim = True)[0]
	data_norm = data_zmean / (data_max + 1e-15)

	return data_norm.view(b, c, d, h, w)


def normalize_feature(data_bxcxdxhxw):
	nn.ReLU()(data_bxcxdxhxw)

	b, c, d, h, w = data_bxcxdxhxw.shape
	data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)

	data_min = data_bxcxk.min(2, keepdim = True)[0]
	data_zmean = data_bxcxk - data_min

	data_max = data_zmean.max(2, keepdim = True)[0]
	data_norm = data_zmean / (data_max + 1e-15)
	data_norm *= 10.

	return data_norm.view(b, c, d, h, w)


class VisibleNet(nn.Module):
	def __init__(self, basedim, layernum=0):
		super().__init__()

		self.layernum = layernum

	def forward(self, x):
		x = nn.ReLU()(x)
		x = normalize(x)
		x = x * 1.e5

		x5 = x

		depdim = x5.shape[2]
		# print(depdim)
		# raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.max(dim=2)
		raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.topk(4, dim=2)
		
		raw_dep_bxcxhxw = depdim - 1 - raw_dep_bxcxhxw.float()
		raw_dep_bxcxhxw = raw_dep_bxcxhxw / (depdim - 1)

		xflatdep = torch.cat([raw_pred_bxcxhxw, raw_dep_bxcxhxw], dim=1)

		return xflatdep





if __name__ == '__main__':
	import os
	import cv2
	import numpy as np
	from scipy.io import loadmat


	# data = loadmat(r'D:\LPP\confocal_nlos_code\data_bunny.mat')
	# rect_data_hxwxt = data['rect_data']

	measFile = r"D:\LPP\nlospose\data\nlospose\train\meas\light-1-411.9427-4.0100-100.0000.hdr"
	meas = cv2.imread(measFile, -1)
	meas = meas / np.max(meas)
	meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
	meas = meas / np.max(meas)
	from einops import rearrange
	meas = rearrange(meas, '(t h) w ->h w t', t=600)

	rect_data_hxwxt = meas[:,:,:512]

	# crop = 1024
	# bin_len = 8e-12 * 3e8
	crop = 512
	bin_len = 0.01

	K = 1
	for k in range(K):
# rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
# rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
		rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]	
		crop = crop // 2
		bin_len = bin_len * 2

	dev = 'cuda'
	sptial_grid = rect_data_hxwxt.shape[0]

	rect_data_hxwxt = np.float32(rect_data_hxwxt)
	rect_data_dxhxwxt = np.expand_dims(rect_data_hxwxt, axis=0)
	rect_data_bxdxhxwxt = np.expand_dims(rect_data_dxhxwxt, axis=0)

	bnum = 1
	dnum = 1
	rect_data_bxdxhxwxt = np.tile(rect_data_bxdxhxwxt, [bnum, dnum, 1, 1, 1])
	rect_data_bxdxhxwxt = torch.from_numpy(rect_data_bxdxhxwxt).cuda()

#####################################################################
	lctlayer = LCT(image_size=sptial_grid, wall_size=2, time_size=crop, bin_len=bin_len,
   		mode='lct')
	lctlayer.todev(dev, dnum)

	temds = False

	tbe = 0 // (2 ** K)
	if temds:
		tlen = 512 // (2 ** K)
	else:
		tlen = 256

	for i in range(10):
		print(i)
	re = lctlayer(rect_data_bxdxhxwxt[:, :, :, :, tbe:tbe + tlen].permute(0, 1, 4, 2, 3), \
  		[tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen])

	print(f'output\'s shape is {re.shape}')
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

	left_view = np.max(volumn_MxNxN, axis = 1)
	cv2.imshow("left", left_view / np.max(left_view))
	cv2.waitKey()

	top_view = np.max(volumn_MxNxN, axis = 2)
	cv2.imshow("top", top_view / np.max(top_view))
	cv2.waitKey()
# volumn_ZxYxX = volumn_MxNxN
# volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
# for i, frame in enumerate(volumn_ZxYxX):
# print(i)
# cv2.imshow("re1", frame)
# cv2.imshow("re2", frame / np.max(frame))
# cv2.waitKey(0)

