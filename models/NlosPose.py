import os
import torch
from torch import nn
from einops import rearrange

from models.feature_extraction import FeatureExtraction
from models.feature_propagation import FeaturePropagation, VisibleNet, normalize_feature
from models.posenet import get_config, get_pose_net
from models.posenet3d_50 import get_pose_net_50
from unet.unet3d import UNet3d, freeze_layer


class NlosPose(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.time_begin = 0
		self.time_end = cfg.MODEL.TIME_SIZE

		self.feature_extraction = FeatureExtraction(
			basedim=cfg.MODEL.BASEDIM,
			in_channels=cfg.MODEL.IN_CHANNELS,
			stride=1
		)
		self.feature_propagation = FeaturePropagation(
			time_size=cfg.MODEL.TIME_SIZE,
			image_size=cfg.MODEL.IMAGE_SIZE[0],
			wall_size=cfg.MODEL.WALL_SIZE,
			bin_len=cfg.MODEL.BIN_LEN,
			dnum=cfg.MODEL.DNUM,
			dev=cfg.DEVICE
		)

		if cfg.MODEL.PRETRAIN_AUTOENCODER == True:
			self.autoencoder = torch.load(cfg.MODEL.PRETRAIN_AUTOENCODER_PATH, map_location=f"cuda:{cfg.DEVICE}")
			# freeze_layer(self.autoencoder)
		else:self.autoencoder = UNet3d(
                in_channels=1,
                n_channels=4,
            )
		if cfg.MODEL.BACKBONE == 'posenet2d':
			self.vis_net = VisibleNet(basedim=3)
			self.pose_net_cfg = get_config()
			self.pose_net = \
				get_pose_net(self.pose_net_cfg, num_joints=cfg.MODEL.NUM_JOINTS)
		elif cfg.MODEL.BACKBONE == 'posenet3d_50':
			self.pose_net = get_pose_net_50()

	def forward(self, meas): # (2,1,128,64,64)

		meas = self.feature_extraction(meas)  # (2,2,64,32,32)
		
		feature = self.feature_propagation(meas, [self.time_begin, self.time_begin, self.time_begin], [self.time_end , self.time_end, self.time_end])  #(2, 1, 128, 64, 64)
		feature = normalize_feature(feature)
		refine_feature = self.autoencoder(feature)

		output = self.pose_net(feature + refine_feature)

		return output, refine_feature


if __name__ == '__main__':
	from config.config_noise import _C as cfg
	model = NlosPose(cfg)
	video = torch.randn(1,1,cfg.MODEL.TIME_SIZE,cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
	pre = model(video)


