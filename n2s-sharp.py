import sys
sys.path.append('..')
from data import N2SDataset
import model

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json

import torch
from torchvision import transforms
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from noise2self.mask import Masker
import model
import progress

"""
Config format:
{
	model: "res"/",
	model_params: {},
	train: {
		batch_size: int,
		epochs: int
	}
}
"""


def cli():
	parser = ArgumentParser()
	parser.add_argument("data", type=str, help="Root folder of the pro data")
	parser.add_argument("--config", type=str, nargs="?", default="denoising-default.json", help="path to configuration file for training setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	


def fit(net, loss_function, dataset, epochs, batch_size=32, device='cpu', mask_grid_size=4, fade_threshold=1000):

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size

	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

	masker = Masker(width = mask_grid_size, mode='interpolate')
	optimizer = Adam(net.parameters(), lr=0.001)

	dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=3)

	net.to(device)
	loss_function.to(device)

	plot_val(net, test_data_loader, device, 0)

	# training
	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=train_size)
		net.train()
		for i, batch in enumerate(dataloader):
			noisy_images = batch.to(device)
			noisy_images = noisy_images.float()

			# for now only low input
			noisy_images = noisy_images[:,:1,::]
			net_input, mask = masker.mask(noisy_images, i)
			net_output = net(net_input)

			fade = transforms.Lambda(lambda x: fading_loss(x, threshold_from_end=fade_threshold))
			fade_factor = fade(noisy_images*mask)

			loss = loss_function(net_output*mask, noisy_images*mask*fade_factor)

			optimizer.zero_grad()

			loss.backward()

			optimizer.step()
			bar.inc_progress(batch_size)

		
		print('\nLoss (', i, ' \t', round(loss.item(), 6))
		with open('loss.txt', 'a') as f:
			print(e, round(loss.item(), 6), '\n', file=f)
   

		net.eval()
		plot_val(net, test_data_loader, device, e)

def plot_val(net, data_loader, device, e):
	i, test_batch = next(enumerate(data_loader))
	noisy = test_batch.to(device)
	noisy = noisy.float()
	denoised = net(noisy[:,1:,::]).detach()
	noisy = noisy[:,1:,::]
	noisy, denoised = noisy.cpu(), denoised.cpu()
	comp = np.concatenate([noisy, denoised], axis=-2)

	n_pics = 5
	fig = plt.figure(figsize=(3*n_pics, 8))
	fig.suptitle('channel low, noisy(top) vs denoised(bottom)', fontsize=30)
	for j in range(n_pics):
		ax1 = fig.add_subplot(1,n_pics,j+1)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title("image {}".format(j))
		ax1.imshow(comp[j][0], interpolation=None, vmin=0.0, vmax=1.0, cmap='gray')

	plt.savefig('n2s_epoch' + str(e) + '.png', dpi=300)


def fading_loss(x, threshold_from_end=1000, maxvalue=65535.0):
	"""
	creates fading factor that fades out linearly from 1 at threshold to 0 at maxvalue
	"""
	if threshold_from_end == 0:
		return torch.full_like(x, 1.0)

	mask = x < (maxvalue - threshold_from_end)
	x =  1.0 - ((x - torch.full_like(x, maxvalue - threshold_from_end)) / threshold_from_end)
	x[mask] = 1.0
	return x
	# simple python version
	# if x > (maxvalue - start):
	# if x < 0:	# if x 		# return (-(x - maxvalue)) / start
	# else:
		# return 1


if __name__=="__main__":
	args = cli()
	
	with open(args.config, 'r') as f:
		config = json.load(f)

	# datasets
	dataset = N2SDataset(args.data, sharp=True, **config.get("dataset", {}))

	simple_res_net = torch.nn.Sequential(
		model.ConvBlock(1, 16, 3, padding_mode='reflect'),
		model.ResBlock(1, 3, padding_mode='reflect', activation='relu', hidden_channels=[32, 32, 32]),
		model.ResBlock(1, 3, padding_mode='reflect', activation='relu', hidden_channels=[32, 32, 32]),
		model.ConvBlock(1, 1, 3, padding_mode='reflect', activation='sigmoid'),
	)

	model_type = config['model']
	if model_type == 'resnet':
		from model import ResNet
		net = ResNet(1, 1, **config.get("model_params", {}))# in, out channels
	if model_type == 'n2s-babyu':
		from noise2self.models.babyunet import BabyUnet
		net = BabyUnet()
	if model_type == 'n2s-unet':
		from noise2self.models.unet import Unet
		net = Unet(**config.get("model_params", {}))
	if model_type == 'n2s-dncnn':
		from noise2self.models.dncnn import DnCNN
		net = DnCNN(1) # number of channels
	if model_type == 'simple_res':
		net = simple_res_net

	net = net.float()

	loss = MSELoss()

	fit(net,
		loss,
		dataset,
		config['train']['epochs'],
		batch_size=config['train']['batch_size'],
		device=args.device,
		mask_grid_size=config['train']['mask_grid_size'],
		fade_threshold=config['train']['fade_threshold']
		)
