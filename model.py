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
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = nn.Sequential(
				ConvBlock(in_channels*2, in_channels, 3, activation=activation),
				ConvBlock(in_channels, out_channels, 3, activation=activation)
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
	def __init__(self, n_channels, n_classes, bilinear=True, activation='relu', hidden=[64, 128, 256, 512]):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.bilinear = bilinear

		self.inconv = nn.Sequential(
			ConvBlock(n_channels, hidden[0], 3, activation=activation),
			ConvBlock(hidden[0], hidden[0], 3, activation=activation)
		)
		self.down_convs = [Down(i, o) for i, o in zip(hidden[:-1], hidden[1:])]
		self.up_convs = [UpSkip(i, o, bilinear) for i, o in zip(reversed(hidden[1:]), reversed(hidden[:-1]))]
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
	def __init__(self, in_channels, kernel_size, padding_mode='zeros', hidden_channels=[], activation='relu', last_layer_activation=False):
		super(ResBlock, self).__init__()
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
		if self.activation is not None:
			out = self.activation(out)
		return x + out
