import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.cuda.comm import broadcast
from torch.autograd import Variable

class NMTNORMCritierion(nn.Module):
	"""docstring for NMTNORMCritierion"""
	def __init__(self, label_smoothing = 0.0):
		super(NMTNORMCritierion, self).__init__()
		self.label_smoothing = label_smoothing
		self.LogSoftmax = nn.LogSoftmax(dim = 1)

		if label_smoothing > 0:
			self.criterion_ = nn.KLDivLoss(reduction = 'none')
		else:
			self.criterion_ = nn.NLLLoss(reduction = 'none',ignore_index = 100000)
		self.confidence = 1.0 - label_smoothing

	def _smooth_label(self, num_tokens):
		one_hot = torch.randn(1, num_tokens)
		one_hot.fill_(self.label_smoothing / (num_tokens - 1))
		return one_hot

	def _bottle(self, v):
		return v.view(-1, v.size(2))

	def criterion(self, dec_outs, labels):
		scores = self.LogSoftmax(dec_outs)
		num_tokens = scores.size(-1)

		gtruth = labels.view(-1)
		if self.confidence < 1:
			tdata = gtruth.detach()
			one_hot = self._smooth_label(num_tokens)
			if labels.is_cuda:
				one_hot = one_hot.cuda()
			tmp_ = one_hot.repeat(gtruth.size(0) ,1)
			tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
			gtruth = tmp_.detach()
		loss = torch.mean(self.criterion_(scores, gtruth), dim = 1)
		return loss

	def forward(self, output_x, output_y, target, target_weight):
		batch_size = output_x.size(0)
		num_joints = output_x.size(1)
		loss = 0 

		for idx in range(num_joints):
			coord_x_pred = output_x[:, idx].squeeze()
			coord_y_pred = output_y[:, idx].squeeze()
			coord_gt = target[:, idx].squeeze()
			weight = target_weight[:, idx].squeeze()

			loss += self.criterion(coord_x_pred, coord_gt[:, 0].mul(weight).mean())
			loss += self.criterion(coord_y_pred, coord_gt[:, 1].mul(weight).mean())

		return loss / num_joints



class L2JointLocationLoss(nn.Module):
	def __init__(self, output_3d, size_average=True, reduce=True):
		super(L2JointLocationLoss, self).__init__()
		self.size_average = size_average
		self.reduce = reduce
		self.output_3d = output_3d

	def forward(self, preds, *args):
		gt_joints = args[0]
		gt_joints_vis = args[1]

		num_joints = int(gt_joints_vis.shape[1] / 3)
		hm_width = preds.shape[-1]
		hm_height = preds.shape[-2]
		# hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1
		hm_depth = preds.shape[-3]

		pred_jts = softmax_integral_tensor(preds, num_joints,  hm_width, hm_height, hm_depth)

		_assert_no_grad(gt_joints)
		_assert_no_grad(gt_joints_vis)
		return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average)


def _assert_no_grad(tensor):
	assert not tensor.requires_grad, \
		"nn criterions don't compute the gradient w.r.t. targets - please " \
		"mark these tensors as not requiring gradients"


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
	assert isinstance(heatmaps, torch.Tensor)
	'''
    Parameter


    heatmaps: probility of location
    -----------------------------------
    Return 
    accu_x: mean location of x label

    '''

	heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

	accu_x = heatmaps.sum(dim=2)  # (1, 24, 5, 5, 5)
	accu_x = accu_x.sum(dim=2)
	accu_y = heatmaps.sum(dim=2)
	accu_y = accu_y.sum(dim=3)
	accu_z = heatmaps.sum(dim=3)
	accu_z = accu_z.sum(dim=3)

	accu_x = accu_x * \
			 torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[
				 0]
	accu_y = accu_y * \
			 torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[
				 0]
	accu_z = accu_z * \
			 torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[
				 0]

	accu_x = accu_x.sum(dim=2, keepdim=True)
	accu_y = accu_y.sum(dim=2, keepdim=True)
	accu_z = accu_z.sum(dim=2, keepdim=True)

	return accu_x, accu_y, accu_z


def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
	# global soft max
	preds = preds.reshape((preds.shape[0], num_joints, -1))
	preds = F.softmax(preds, 2)

	# integrate heatmap into joint location
	if output_3d:
		x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
	else:
		assert 0, 'Not Implemented!'  # TODO: Not Implemented
	x = x / float(hm_width) - 0.5
	y = y / float(hm_height) - 0.5
	z = z / float(hm_depth) - 0.5
	preds = torch.cat((x, y, z), dim=2)
	preds = preds.reshape((preds.shape[0], num_joints * 3))
	return preds


def weighted_mse_loss(input, target, weights, size_average):
	out = (input - target) ** 2
	out = out * weights
	if size_average:
		return out.sum() / len(input)
	else:
		return out.sum()




if __name__ == '__main__':

	test_NMTNORMCritierion = False
	test_L2JointLocationLoss = True

	if test_NMTNORMCritierion:
		criterion = nn.KLDivLoss(reduction = 'mean')
		output = torch.Tensor([[0.1132, 0.5477, 0.3390]])
		target = torch.Tensor([[0.8541, 0.0511, 0.0947]])
		# output = F.log_softmax(output, dim = 1)
		# target = F.log_softmax(target, dim = 1)
		loss = criterion(output, target)
		print(loss)
		import numpy as np

		output = output[0].detach().numpy()
		output_1 = output[0]
		target_1 = target[0][0].numpy()

		loss_1 = target_1 * (np.log(target_1) - output_1)
		print(loss_1)

	if test_L2JointLocationLoss:
		batch_size = 2
		num_joints = 24
		z_dim = 5
		y_dim = 5
		x_dim = 5

		input = torch.zeros(batch_size, num_joints, z_dim, y_dim, x_dim).cuda()
		# input[0, 0, 2, 2, 2] = 1.
		input[0, 1, 3, 3, 3] = 1.
		print(input.shape)
		# test_a = generate_3d_integral_preds_tensor(input, num_joints, z_dim, y_dim, x_dim)
		# test_b = softmax_integral_tensor(input, 24, True, 5, 5, 5)

		Loss = L2JointLocationLoss(output_3d=True)
		gt_joints = torch.zeros(batch_size, num_joints, 3).cuda()
		gt_joints[0, 1] = -0.1
		gt_joints_vis = torch.ones_like(gt_joints).cuda()
		# gt_joints = rearrange(gt_joints, 'b n d -> b (n d)')
		gt_joints = gt_joints.reshape((batch_size, num_joints * 3))
		gt_joints_vis = rearrange(gt_joints_vis, 'b n d -> b (n d)')
		print(gt_joints[0, 0])
		loss = Loss(input, gt_joints, gt_joints_vis)
		print('finished')