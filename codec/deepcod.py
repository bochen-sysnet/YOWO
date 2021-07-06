import cv2
import numpy as np
import time,sys
import torch
import glob
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
sys.path.append('..')
from codec.huffman import HuffmanCoding


class Middle_conv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Middle_conv, self).__init__()
		self.bn = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv = spectral_norm(self.conv)

	def forward(self, x):
		x = self.conv(F.relu(self.bn(x)))

		return x

class DeepCOD(nn.Module):

	def __init__(self, kernel_size=4, num_centers=8):
		super(DeepCOD, self).__init__()
		out_size = 3
		no_of_hidden_units = 64
		self.encoder = LightweightEncoder(out_size, kernel_size=4, num_centers=8)
		self.conv1 = Middle_conv(out_size,out_size)
		self.resblock_up1 = Resblock_up(out_size,no_of_hidden_units)
		self.conv2 = Middle_conv(no_of_hidden_units,no_of_hidden_units)
		self.resblock_up2 = Resblock_up(no_of_hidden_units,no_of_hidden_units)
		self.output_conv = Output_conv(no_of_hidden_units)
		
	def forward(self, x):
		x,r,t = self.encoder(x)

		# reconstruct
		x = self.conv1(x)
		x = self.resblock_up1(x)
		x = self.conv2(x)
		x = self.resblock_up2(x)
		x = self.output_conv(x)
		
		return x,r,t
def orthorgonal_regularizer(w,scale,cuda=False):
	N, C, H, W = w.size()
	w = w.view(N*C, H, W)
	weight_squared = torch.bmm(w, w.permute(0, 2, 1))
	ones = torch.ones(N * C, H, H, dtype=torch.float32)
	diag = torch.eye(H, dtype=torch.float32)
	tmp = ones - diag
	if cuda:tmp = tmp.cuda()
	loss_orth = ((weight_squared * tmp) ** 2).sum()
	return loss_orth*scale

class Attention(nn.Module):

	def __init__(self, channels, hidden_channels):
		super(Attention, self).__init__()
		f_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.f_conv = spectral_norm(f_conv)
		g_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.g_conv = spectral_norm(g_conv)
		h_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.h_conv = spectral_norm(h_conv)
		v_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.v_conv = spectral_norm(v_conv)
		self.gamma = torch.nn.Parameter(torch.FloatTensor([0.0]))
		self.hidden_channels = hidden_channels
		self.channels = channels

	def forward(self,x):
		nb, nc, imgh, imgw = x.size() 

		f = (self.f_conv(x)).view(nb,self.hidden_channels,-1)
		g = (self.g_conv(x)).view(nb,self.hidden_channels,-1)
		h = (self.h_conv(x)).view(nb,self.hidden_channels,-1)

		s = torch.matmul(f.transpose(1,2),g)
		beta = F.softmax(s, dim=-1)
		o = torch.matmul(beta,h.transpose(1,2))
		o = self.v_conv(o.transpose(1,2).view(nb,self.hidden_channels,imgh,imgw))
		x = self.gamma * o + x

		return x

class Resblock_up(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Resblock_up, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		# self.relu1 = nn.LeakyReLU()
		deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv1 = spectral_norm(deconv1)

		self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
		# self.relu2 = nn.LeakyReLU()
		deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
		self.deconv2 = spectral_norm(deconv2)

		self.bn3 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		# self.relu3 = nn.LeakyReLU()
		deconv_skip = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv_skip = spectral_norm(deconv_skip)

	def forward(self, x_init):
		x = self.deconv1(F.relu(self.bn1(x_init)))
		x = self.deconv2(F.relu(self.bn2(x)))
		x_init = self.deconv_skip(F.relu(self.bn3(x_init)))
		return x + x_init

class LightweightEncoder(nn.Module):

	def __init__(self, channels, kernel_size=4, num_centers=8):
		super(LightweightEncoder, self).__init__()
		self.sample = nn.Conv2d(3, channels, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True)
		self.sample = spectral_norm(self.sample)
		self.centers = torch.nn.Parameter(torch.rand(num_centers))
		self.pool1 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
		self.unpool = nn.Upsample(scale_factor=2, mode='nearest')

	def forward(self, x):
		start = time.perf_counter()
		x = self.sample(x)
		B,C,H,W = x.size()

		# quantization
		xsize = list(x.size())
		x = x.view(*(xsize + [1]))
		quant_dist = torch.pow(x-self.centers, 2)
		softout = torch.sum(self.centers * nn.functional.softmax(-quant_dist, dim=-1), dim=-1)
		minval,index = torch.min(quant_dist, dim=-1, keepdim=True)
		hardout = torch.sum(self.centers * (minval == quant_dist), dim=-1)
		x = softout
		huffman = HuffmanCoding()
		real_size = len(huffman.compress(index.view(-1).cpu().numpy())) * 4
		dur = time.perf_counter() - start
		real_cr = 1/16.*real_size/(H*W*C*B*8)
		return x,real_cr,dur

def mask_compression(mask):
	prev = 1
	rl = 0
	cnt = 0
	result = []
	for e in mask:
		if e == prev:
			rl += 1
		else:
			result += [rl]
			rl = 0
		prev = e
	if rl>0:
		result += [rl]
	huffman = HuffmanCoding()
	size = len(huffman.compress(result))*4
	return size

class Output_conv(nn.Module):

	def __init__(self, channels):
		super(Output_conv, self).__init__()
		self.bn = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		# self.relu = nn.LeakyReLU()#nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv = spectral_norm(self.conv)

	def forward(self, x):
		x = self.conv(F.relu(self.bn(x)))
		x = torch.tanh(x)
		x = (x+1)/2

		return x

def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')
		# nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
	pass
	