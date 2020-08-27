import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

_activations = {
	'relu': torch.nn.ReLU,
	'tanh': torch.nn.Tanh,
	'sigmoid': torch.nn.Sigmoid,
	'leakyrelu': torch.nn.LeakyReLU
}


class Sobel(nn.Module):
	def __init__(self, n_channels):
		super(Sobel, self).__init__()
		self.sobel_x = np.array([[[[1, 0, -1],[2,0,-2],[1,0,-1]]] * n_channels])
		self.sobel_y = np.array([[[[1, 2, 1],[0,0,0],[-1,-2,-1]]] * n_channels])
		self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv_x.weight=nn.Parameter(torch.from_numpy(self.sobel_x).float())
		self.conv_y.weight=nn.Parameter(torch.from_numpy(self.sobel_y).float())

	def forward(self, x):
		G_x=self.conv_x(x)
		G_y=self.conv_y(x)

		return torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))


class SmoothMSELoss(nn.Module):
	def __init__(self, n_channels, alpha=0.1):
		super(SmoothMSELoss, self).__init__()
		self.mse = torch.nn.MSELoss()
		self.sobel = Sobel(n_channels)
		self.alpha = alpha
	
	def forward(self, prediction, target):
		sobel_target = self.sobel(target)
		sobel_prediction = self.sobel(prediction)
		smooth = sobel_prediction * torch.exp(-sobel_target) 
		return self.mse(prediction, target) + self.alpha * torch.mean(smooth)


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding_mode='zeros', activation='relu'):
		super(ConvBlock, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding_mode=padding_mode, padding=kernel_size//2)
		self.bn = torch.nn.BatchNorm2d(in_channels)
		self.activation = _activations[activation]()

	def forward(self, x):
		return self.activation(self.conv(self.bn(x)))


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, activation='relu'):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			ConvBlock(in_channels, out_channels, 3, activation=activation),
			ConvBlock(out_channels, out_channels, 3, activation=activation)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class UpSkip(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True, activation='relu'):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1)
			)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
		
		self.conv = nn.Sequential(
			ConvBlock(in_channels, out_channels, 3, activation=activation),
			ConvBlock(out_channels, out_channels, 3, activation=activation)
		)

	def forward(self, x, skip):
		x = self.up(x)
		# input is CHW
		diffY = skip.size()[2] - x.size()[2]
		diffX = skip.size()[3] - x.size()[3]

		x = F.pad(x, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


class UNet(nn.Module):
	def __init__(self, n_channels, bilinear=True, activation='relu', hidden=[64, 128, 256, 512], padding_mode='reflect'):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.bilinear = bilinear

		self.inconv = nn.Sequential(
			ConvBlock(n_channels, hidden[0], 3, activation=activation, padding_mode=padding_mode),
			ConvBlock(hidden[0], hidden[0], 3, activation=activation, padding_mode=padding_mode)
		)
		self.down_convs = nn.ModuleList([Down(i, o) for i, o in zip(hidden[:-1], hidden[1:])])
		self.up_convs = nn.ModuleList([UpSkip(i, o, bilinear) for i, o in zip(reversed(hidden[1:]), reversed(hidden[:-1]))])
		self.outconv = nn.Conv2d(hidden[0], n_channels, kernel_size=1)

	def forward(self, x):
		x = [self.inconv(x)]
		for down in self.down_convs:
			x.append(down(x[-1]))

		out = x[-1]
		for up, skip in zip(self.up_convs, reversed(x[:-1])):
			out = up(out, skip)
		
		return self.outconv(out)

class ResBlock(nn.Module):
	def __init__(self, in_channels, kernel_size, padding_mode='zeros', hidden_channels=[], activation='relu', last_layer_activation=True):
		super(ResBlock, self).__init__()
		self.last_layer_activation = last_layer_activation
		self.activation = _activations[activation]() if last_layer_activation else None
		if len(hidden_channels) == 0:
			hidden_channels = [in_channels]
		self.convs = torch.nn.ModuleList([ConvBlock(i, o, kernel_size, activation=activation) for i, o in zip([in_channels]+hidden_channels[:-1], hidden_channels)])
		self.conv_out = torch.nn.Conv2d(hidden_channels[-1], in_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
		

	def forward(self, x):
		out = x
		for conv in self.convs:
			out = conv(out)
		out = self.conv_out(out)
		if self.last_layer_activation:
			out = self.activation(out)
		return x + out


class ResNet(nn.Module):
	"""Stack multiple resblocks and an in and out convolutiono"""
	def __init__(self, in_channels, out_channels, in_conv=[16, 32, 64], res_blocks=[[64, 64, 64], [64, 64, 64], [64, 64, 64]], activation='relu', full_res=False, last_layer_activation='none', padding_mode='zeros'):
		super(ResNet, self).__init__()
		self.net = torch.nn.Sequential(
			*[ConvBlock(i, o, 3, activation=activation, padding_mode=padding_mode) for i, o in zip([in_channels]+in_conv[:-1], in_conv)],
			*[ResBlock(in_conv[-1], 3, hidden_channels=block, activation=activation, padding_mode=padding_mode) for block in res_blocks],
			nn.Conv2d(in_conv[-1], in_channels, kernel_size=1)
		)
		self.full_res = full_res
		if last_layer_activation=='none':
			self.out_activation = None
		elif last_layer_activation=='sigmoid':
			self.out_activation = torch.nn.Sigmoid()
		elif last_layer_activation=='relu':
			self.out_activation = torch.nn.ReLU()
	
	def forward(self, x):
		if self.full_res:
			out = self.net(x) + x
		else:
			out = self.net(x)
		
		if self.out_activation:
			return self.out_activation(out)
		
		return out