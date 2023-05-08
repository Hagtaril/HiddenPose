import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class FeatureExtracion_UNet(nn.Module):
	"""docstring for FeatureExtracion_UNet"""

	def __init__(self, cfg, in_channels, out_channels, features=[128, 256, 512, 1024]):
		super(FeatureExtracion_UNet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.features = features
		self.down = UNET_DOWN(self.in_channels)
		self.up = UNET_UP(self.in_channels, self.out_channels)

	def forward(self, x):
		x, skip_x = self.down(x)
		output = self.up(x, skip_x)
		return output


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.conv(x)


class UNET_DOWN(nn.Module):
	def __init__(
			self, in_channels=3, features=[128, 256, 512, 1024],
	):
		super(UNET_DOWN, self).__init__()
		# self.ups = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Down part of UNET
		for feature in features:
			self.downs.append(DoubleConv(in_channels, feature))
			in_channels = feature

		# Up part of UNET
		# for feature in reversed(features):
		#     self.ups.append(
		#         nn.ConvTranspose2d(
		#             feature*2, feature, kernel_size=2, stride=2,
		#         )
		#     )
		#     self.ups.append(DoubleConv(feature*2, feature))

		self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

	# self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

	def forward(self, x):
		skip_connections = []

		for down in self.downs:
			x = down(x)
			skip_connections.append(x)
			x = self.pool(x)

		x = self.bottleneck(x)
		skip_connections = skip_connections[::-1]

		return x, skip_connections


class UNET_UP(nn.Module):
	def __init__(
			self, in_channels=3, out_channels=1, features=[128, 256, 512, 1024],
	):
		super(UNET_UP, self).__init__()
		self.ups = nn.ModuleList()
		# self.downs = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Down part of UNET
		# for feature in features:
		#     self.downs.append(DoubleConv(in_channels, feature))
		#     in_channels = feature

		# Up part of UNET
		for feature in reversed(features):
			self.ups.append(
				nn.ConvTranspose2d(
					feature * 2, feature, kernel_size=2, stride=2,
				)
			)
			self.ups.append(DoubleConv(feature * 2, feature))

		# self.bottleneck = DoubleConv(features[-1], features[-1]*2)
		self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

	def forward(self, x, skip_connections):
		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_connection = skip_connections[idx // 2]

			if x.shape != skip_connection.shape:
				x = TF.resize(x, size=skip_connection.shape[2:])

			concat_skip = torch.cat((skip_connection, x), dim=1)
			x = self.ups[idx + 1](concat_skip)

		return self.final_conv(x)


class FeatureExtraction(nn.Module):
	"""docstring for FeatureExtraction
	Input: Tensor (batch_size, channels, time_size, image_size[0], image_size[1])
	------------
	Output: Tensor (batch_size, channels, time_size, image_size[0], image_size[1]
	"""
	def __init__(
		self, 
		basedim,
		in_channels,
		stride = 2,
		norm = nn.InstanceNorm3d
	):
		super(FeatureExtraction, self).__init__()
		assert in_channels == 1, \
			f'input channels should be 1, not {in_channels}'
		# TODO simplify
		self.stride = stride

		weights = np.zeros((1,1,3,3,3), dtype = np.float32)
		weights[:,:,1:,1:,1:] = 1.0
		tfweights = torch.from_numpy(weights / np.sum(weights))
		tfweights.requires_grad = True
		self.weights = nn.Parameter(tfweights)

		self.conv1 = nn.Sequential(
			nn.ReplicationPad3d(1),
			nn.Conv3d(
				in_channels,
				basedim,
				kernel_size = [3,3,3],
				padding = 0,
				stride = self.stride,
				bias = True),
			ResConv3D(basedim, inplace = False),
			ResConv3D(basedim, inplace = False)
			)

	def forward(self, x):
		'''
		x : [0,1]

		input: (1, 1, 128, 256, 256)
		'''
		x_conv1 = self.conv1(x)
		x_conv2 = F.conv3d(x, self.weights, bias = None, stride = self.stride, padding = 1, dilation = 1, groups = 1)

		# output = torch.cat([x_conv2, x_conv1], dim = 1)
		output = x_conv1 + x_conv2
		return output




class Conv2(nn.Module):
	"""docstring for FeatureExtraction
	Input: Tensor (batch_size, channels, time_size, image_size[0], image_size[1])
	------------
	Output: Tensor (batch_size, channels, time_size, image_size[0], image_size[1]
	"""
	def __init__(
		self,
		basedim,
		in_channels,
		stride = 1,
		norm = nn.InstanceNorm3d
	):
		super(Conv2, self).__init__()
		assert in_channels == 2, \
			f'input channels should be 1, not {in_channels}'
		# TODO simplify
		self.stride = stride

		weights = np.zeros((24,2,3,3,3), dtype = np.float32)
		weights[:,:,1:,1:,1:] = 1.0
		tfweights = torch.from_numpy(weights / np.sum(weights))
		tfweights.requires_grad = True
		self.weights = nn.Parameter(tfweights)

		self.conv1 = nn.Sequential(
			nn.ReplicationPad3d(1),
			nn.Conv3d(
				in_channels,
				basedim,
				kernel_size = [3,3,3],
				padding = 0,
				stride = self.stride,
				bias = True),
			ResConv3D(basedim, inplace = False),
			ResConv3D(basedim, inplace = False)
			)

	def forward(self, x):
		'''
		x : [0,1]

		input: (2, 2, 128, 256, 256)
		'''
		x_conv1 = self.conv1(x)
		x_conv2 = F.conv3d(x, self.weights, bias = None, stride = self.stride, padding = 1, dilation = 1, groups = 1)

		# output = torch.cat([x_conv2, x_conv1], dim = 1)
		output = x_conv1 + x_conv2
		return output


class ResConv3D(nn.Module):
	"""docstring for ResConv3D"""
	def __init__(self, basedim, inplace = False):
		super(ResConv3D, self).__init__()
		self.inplace = inplace
		self.tmp = nn.Sequential(
			nn.ReplicationPad3d(1),
			nn.Conv3d(
				basedim,
				basedim,
				kernel_size = [3,3,3],
				padding = 0,
				stride = [1,1,1],
				bias = True),
			nn.LeakyReLU(negative_slope = 0.2, inplace = inplace),
			nn.ReplicationPad3d(1),
			nn.Conv3d(
				basedim,
				basedim,
				kernel_size=[3,3,3],
				padding = 0,
				stride = [1,1,1],
				bias = True),
		)

	def forward(self, x):
		x = self.tmp(x) + x
		x = F.leaky_relu(x, negative_slope = 0.2, inplace = self.inplace)
		return x


def test():
    x = torch.randn((2, 128, 64, 64))
    # model_1 = UNET_DOWN(in_channels=128)
    # model_2 = UNET_UP(in_channels = 128, out_channels = 8)
    # x_mid, skip_connections = model_1(x)
    # preds_2 = model_2(x_mid, skip_connections)
    # # assert preds_2.shape == x.shape
    # print(model_1)
    # print(preds_2.shape)
    model = FeatureExtracion_UNet(1, in_channels = 128, out_channels = 8)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
	test_Unet = True
	if not test_Unet:
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

		downnet = FeatureExtraction(basedim=basedim, in_channels=channel)
		downnet = downnet.to(dev)
		tfdata = torch.from_numpy(data).to(dev)
		tfre = downnet(tfdata)
	# tfre = nn.ConstantPad3d((0, 0, 0, 0, 2, 3), 0)(tfre)
		print('\n')
		print(tfre.shape)

	if test_Unet:
		test()