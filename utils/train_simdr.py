import time
import torch
from torch import nn
from einops import rearrange



def train_simdr(
	cfg,
	train_loader, 
	model, 
	criterion, 
	optimizer, 
	epoch,
	output_dir,
	writer,
	# tb_log_dir,
	# writer_dict
):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()

	model.train()

	end = time.time()
	for step, (input, target) in enumerate(train_loader):
		target_weight = torch.ones_like(target[:,:,0])
		data_time.update(time.time() - end)

		input = input.to(cfg.DEVICE)
		target = target.to(cfg.DEVICE)

		output = model(input)
		# output_x, output_y, output_z = map(lambda t:output[:,:,t,:], (0,1,2))

		# target = rearrange(target, 'b n a -> b (n a)' )
		target_weight = torch.ones_like(target)

		target = target.cuda(non_blocking = True).float()
		target_weight = target_weight.cuda(non_blocking = True).float()


		# output = torch.mean(output, dim = 3)
		global_iter_num = epoch * len(train_loader) + step + 1
		
		criterion_test = nn.MSELoss()
		# loss = criterion(output, target, target_weight, writer, global_iter_num)
		# loss = criterion(output_x, output_y, output_z, target, target_weight)
		gt_joints = torch.zeros_like(output)
		for idx in range(target.shape[1]):
			x = target[:,idx,0]
			y = target[:,idx,1]
			z = target[:,idx,2]
			gt_joints[:,idx,z.long(),x.long(),y.long()] = 10
		# loss = criterion(output, target, target_weight)
		loss = criterion_test(output, gt_joints)
		print(f'loss is {loss.item()}')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), input.size(0))
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, global_iter_num)
		batch_time.update(time.time() - end)
		end = time.time()

		


class AverageMeter(object):
	"""docstring for AverageMeter"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n = 1):
		self.val = val
		self.sum += val * n 
		self.count += n 
		self.avg = self.sum / self.count if self.count != 0 else 0
		

if __name__ == '__main__':
	pass