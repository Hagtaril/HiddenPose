import numpy as np
import torch
import torch.nn as nn

class JointsRepresentation(nn.Module):
	"""docstring for JointsRepresentation"""
	def __init__(self):
		super(JointsRepresentation, self).__init__()
		
		self.image_size = torch.Tensor([5,5])
		self.heatmap_size = torch.Tensor([5,5])
		self.num_joints = torch.Tensor(1)
		self.sigma = torch.Tensor(1) #TODO
		self.target_type = 'gaussian'
		
	def generate_target(self, joints, joints_vis):
		target_weight = np.ones((self.num_joints, 1), dtype = np.float32)
		target_weight[:, 0] = joints_vis[:, 0]

		assert self.target_type == 'gaussian', \
			'Only support gaussian'

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
				# Check that any part of the gaussian is in_bounds
				ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
				br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
				if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
						or br[0] < 0 or br[1] < 0:
					# If not, just return the image as is
					target_weight[joint_id] = 0
					continue

				# Generate gaussian
				size = 2 * tmp_size + 1
				x = np.arange(0, size, 1, np.float32)
				y = x[:, np.newaxis]
				x0 = y0 = size // 2 
				# normalize
				g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

				# Usable gaussian range
				g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
				g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

				# Image range
				img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
				img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

				v = target_weight[joint_id]
				if v > 0.5:
					target[joint_id][img_y[0]:img[1], img_x[0], img[0]:img_x[1]] = \
						g[g_y[0]:g_y[1], g_x[0]: g_x[1]]

		if self.use_different_joints_weight:
			target_weight = np.multiply(target_weight, self.joints_weight)

		return target, target_weight



if __name__ == '__main__':
	joint_heatmap = JointsRepresentation()
	img = torch.ones(5,5)
	joints = torch.Tensor([[2,2,0]])
	joints_vis = torch.ones_like(joints)
	target , target_weight = joint_heatmap.generate_target(joints, joints_vis)
	print(target.shape)
	print(target)
