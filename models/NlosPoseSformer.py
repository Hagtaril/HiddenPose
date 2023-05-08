from math import sqrt
import numpy as np
from einops import rearrange, repeat
from math import log, pi
import torch
from torch import nn, einsum
import torch.nn.functional as F
from timm.models.layers.weight_init import trunc_normal_


class NlosPoseSformer(nn.Module):
	"""docstring for NlosPoseSformer
	INPUT :  Tensor (batch_size, channels, time_size, image_size[0], image_size[1])
	--------------
	OUTPUT: Tensor (batch_size, num_joints, dim_joints)
	"""

	def __init__(
			self,
			*,
			dim,
			num_frames,
			num_joints=24,
			image_size=224,
			patch_size=16,
			channels=2,
			depth=12,
			heads=8,
			dim_head=64,
			attn_dropout=0.,
			ff_dropout=0.,
			rotary_emb=True,
			out_dim= 64 * 2 * 3,
			batch_size = 2,# output shape
			# shift_tokens = False
	):
		super(NlosPoseSformer, self).__init__()
		print(type(image_size))
		assert image_size % patch_size == 0, \
			'Image dimensions must be divisible by the patch size.'

		num_patches = (image_size // patch_size) ** 2
		num_positions = num_frames * num_patches
		patch_dim = channels * patch_size ** 2

		self.heads = heads
		self.patch_size = patch_size
		self.num_joints = num_joints
		# self.to_patch_embedding = nn.Linear(patch_dim, dim)
		self.to_patch_embedding = nn.Linear(patch_dim, dim)
		self.joints_token = nn.Parameter(torch.zeros(1, self.num_joints, dim))
		self.to_joints_token = nn.Identity()
		trunc_normal_(self.joints_token, std = .02)

		self.use_rorary_emb = rotary_emb
		if rotary_emb:
			self.frame_rot_emb = RotaryEmbedding(dim_head)
			self.image_rot_emb = AxialRotaryEmbedding(dim_head)
		else:
			self.pose_emb = nn.Embedding(num_positions + 1, dim)

		self.layers = nn.ModuleList([])
		self.to_keypoint_token = nn.Identity()
		for _ in range(depth):
			ff = FeedForward(dim, dropout=ff_dropout)
			time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
			spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

			time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

			self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

		self.to_out = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, out_dim)
		)
		# mid_dim = dim * int(sqrt(out_dim // dim))
		# self.to_out = nn.Sequential(
		# 	nn.LayerNorm(dim),
		# 	nn.Linear(dim, mid_dim),
		# 	nn.LayerNorm(mid_dim),
		# 	nn.Linear(mid_dim, out_dim)
		# )
		# self.to_out_MLP = nn.Sequential(
		# 	nn.LayerNorm(dim),
		# 	nn.Linear(dim, out_dim // 64)  # TODO
		# )
		# assert out_dim % dim == 0, f'out_dim = {out_dim} shold be common multiple of dim = {dim}'
		# mid_factor = int(sqrt(out_dim // dim))
		# self.to_out_CNN = nn.Sequential(
		# 	nn.Conv2d(1, 8, kernel_size = 3, stride = 1, padding = 1),
		# 	nn.Conv2d(8, 64, kernel_size = 3, stride = 1, padding = 1)
		# )

	def forward(self, video, mask=None):
		b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
		assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

		# calculate num patches in height and width dimension, and number of total patches (n)

		hp, wp = (h // p), (w // p)
		n = hp * wp

		# video to patch embeddings

		video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p) #(2, 256, 1, 128, 128) --> (2, 16384 (256 * 8 * 8) , 256 (16 * 16) )
		tokens = self.to_patch_embedding(video)

		joints_token = repeat(self.joints_token, '() n d -> b n d', b=b)  # TODO
		# joints_token = self.joints_token.unsqueeze(0).repeat(b, 1, 1)
		x = torch.cat((joints_token, tokens), dim=1) #(2, 16384 (256 * 8 * 8) , 512 (16 * 16 * 2) ) --> (2, 16408 (24 + 16384), 512)

		frame_pos_emb = None
		image_pos_emb = None
		if not self.use_rorary_emb:
			x += self.pose_emb(torch.arange(x.shape[1], device=device))
		else:
			frame_pos_emb = self.frame_rot_emb(f, device=device)
			image_pos_emb = self.image_rot_emb(hp, wp, device=device)

		frame_mask = None
		cls_attn_mask = None
		if exists(mask):
			mask_with_cls = F.pad(mask, (1, 0), value=True)

			frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n=n, h=self.heads)

			cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n=n, h=self.heads)
			cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value=True)

		x_out = None
		for (time_attn, spatial_attn, ff) in self.layers:
			# x = x + time_attn(x, 'b (f n) d', '(b n) f d', n=n, mask=frame_mask, joints_mask=cls_attn_mask,  #(2, 16408 (24 + 16384), 512) --> ( 128(2 * 64), 256, 512)
			# 				  rot_emb=frame_pos_emb)
			x = x + spatial_attn(x, 'b (f n) d', '(b f) n d', f=f, joints_mask=cls_attn_mask, rot_emb=image_pos_emb)
			x = x + ff(x)
			# if x_out is None:
			# 	x_out = self.to_keypoint_token(x[:, :self.joints_token.shape[1]])
			# else:
			# 	x_out = torch.cat( (x_out, self.to_keypoint_token(x[:, :self.joints_token.shape[1]]) ), dim = 2 )


		joints_token = x[:, :self.joints_token.shape[1]]

		output = rearrange(self.to_out(joints_token), 'b n (p d) -> b n p d', p=4)
		# output = rearrange(self.to_out(x_out), 'b n (p d) -> b n p d', p=4)
		# output_x = self.to_out_x(x_out)
		# output_y = self.to_out_y(x_out)
		# output_z = self.to_out_z(x_out)
		# output = rearrange(self.to_out(x_out), 'b n (p d) -> b n p d', p=4)
		return output





def exists(val):
	return val is not None 

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

def attn(q, k, v, mask = None):
	sim = einsum('b i d, b j d -> b i j', q, k)

	# if exists(mask):
		# max_neg_value = -torch.finfo(sim.dtype).max
		# sim.masked_fill_(~mask, max_neg_value)

	attn = sim.softmax(dim = -1) #[16, 24, 32792]
	out = einsum('b i j, b j d -> b i d', attn , v) # [16, 32792, 32]
	return out #[16, 24, 32]

class PreNorm(nn.Module):
	"""docstring for PreNorm"""
	def __init__(self, dim, fn):
		super(PreNorm, self).__init__()
		self.fn = fn 
		self.norm = nn.LayerNorm(dim)

	def forward(self, x, *args, **kwargs):
		x = self.norm(x)
		return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
	"""docstring for GEGLU"""
	def forward(self, x):
		x, gates = x.chunk(2, dim = -1)
		return x * F.gelu(gates)


class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self, dim, mult = 4, dropout = 0.):
		super(FeedForward, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, dim * mult * 2),
			GEGLU(),
			nn.Dropout(dropout),
			nn.Linear(dim * mult, dim)
		)

	def forward(self, x):
		return self.net(x)


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()


class Attention(nn.Module):
	"""docstring for Attention"""
	def __init__(
		self,
		dim,
		dim_head,
		heads = 8,
		dropout = 0.
	):
		super(Attention, self).__init__()
		self.heads = heads
		self.scale = dim_head ** -0.5
		inner_dim = dim_head * heads

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x, einops_from, einops_to, mask = None,  joints_mask = None, rot_emb = None, num_joints = 24, **einops_dims):
		h = self.heads
		q, k, v = self.to_qkv(x).chunk(3, dim = -1)
		q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

		q = q * self.scale 

		(joints_q, q_), (joints_k, k_), (joints_v, v_) = map(lambda t : (t[:, :num_joints], t[:, num_joints:]), (q, k, v))
         # [16, 24, 32]  , [16, 32768, 32]
		# joints_q = attn(joints_q, joints_k, joints_v) # self attention

		joints_out = attn(joints_q, k, v, mask = joints_mask)
		# [16, 24, 32]
		# ‘b (n f) d -> (b n) f d’
		q_, k_, v_  = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_)) # 'b (f n) d -> (b n) f d'
		# [4096, 128, 32]
		if exists(rot_emb):
			q_, k_ = apply_rot_emb(q_, k_, rot_emb)


		assert q_.shape[0] % joints_k.shape[0] == 0, f'q_.shape[0] % joints_k.shape[0] != 0'
		r = q_.shape[0] // joints_k.shape[0]
		joints_k, joints_v = map(lambda t : repeat(t, 'b n d -> (b r) n d', r = r), (joints_k, joints_v))
		# [4096, 24, 32]
		k_ = torch.cat((joints_k, k_), dim = 1) # [4096, 152, 32]
		v_ = torch.cat((joints_v, v_), dim = 1) # [4096, 152, 32]

		out = attn(q_, k_, v_, mask = mask)

		out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

		out = torch.cat((joints_out, out), dim = 1)

		out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

		return self.to_out(out)






if __name__ == '__main__':
	dim=24 * 3
	image_size=224
	patch_size=16
	num_frames=8
	num_classes=24
	depth=12
	heads=8
	dim_head=64
	attn_dropout=.1
	ff_dropout=.1

	model = NlosPoseSformer(
		dim=dim,
		image_size=image_size,
		patch_size=patch_size,
		num_frames=num_frames,
		num_joints=num_classes,
		depth=depth,
		heads=heads,
		dim_head=dim_head,
		attn_dropout=attn_dropout,
		ff_dropout=ff_dropout
	)

	video = torch.randn(1, 20, 1, 224, 224)

	mask = torch.ones(1, 20).bool()

	pred = model(video, mask = mask)

	print(pred.shape)
	print('NlosPoseSformer finished')







		
		
		