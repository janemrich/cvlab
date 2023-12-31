import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from noise2self.mask import Masker

import model
import progress
from eval import plot_denoise
from eval import plot_denoising_masking
from eval import evaluate_joint
from utils import correct_loss

def fit(net, loss_function, dataset, epochs, target_size, batch_size=32, device='cpu', name=None, mask_grid_size=4, fade_threshold=0, channels=2, learn_rate=0.0001, **kwargs):
	logdir = os.path.join('runs', name + datetime.now().strftime("_%d%b-%H%M%S"))
	writer = SummaryWriter(log_dir=logdir)

	valdir = os.path.join(writer.log_dir, "val")
	os.mkdir(valdir)

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

	test_size = int(0.5 * len(test_dataset))
	val_size = len(test_dataset) - test_size

	test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, val_size], generator=torch.Generator().manual_seed(42))

	dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
	test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

	optimizer_args_default = {'lr': learn_rate * (batch_size/32)}
	optimizer_args = {k[len("optimizer_"):]: v for k, v in kwargs.items() if k.startswith("optimizer_")}
	
	scheduler_args_default = {"patience":1, "cooldown":4}
	scheduler_args = {k[len("scheduler_"):]: v for k, v in kwargs.items() if k.startswith("scheduler_")}
	
	optimizer = torch.optim.Adam(net.parameters(), **{**optimizer_args_default, **optimizer_args})
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{**scheduler_args_default, **scheduler_args})

	net.to(device)
	loss_function.to(device)

	# plot_denoise(net, test_data_loader, device, 0, channels)

	# training
	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=train_size)
		net.train()

		train_loss = 0.0
		n_losses = 0
		for noisy, net_input, mask in dataloader:
			noisy, net_input, mask = noisy.to(device), net_input.to(device), mask.to(device)

			# plot_denoising_masking(noisy, net_input, mask, torch.zeros_like(noisy))
			net_output = net(net_input)

			# plot_denoising_masking(noisy, net_input, mask, net_output)

			# fade = transforms.Lambda(lambda x: fading_loss(x, threshold_from_end=fade_threshold))
			# fade_factor = fade(noisy*mask)

			# loss = loss_function(net_output*mask*fade_factor, noisy*mask*fade_factor)
			loss = loss_function(net_output*mask, noisy*mask)
			loss = loss * correct_loss(mask)
			train_loss += loss.item()
			n_losses += 1

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			writer.add_scalar('Loss/train', loss.item(), global_step=e)
			bar.inc_progress(batch_size)

		train_loss /= n_losses
		del noisy, net_input, mask, net_output, loss, #fade_factor

		net.eval()
		val_loss = 0.0
		n_losses = 0
		for noisy, net_input, mask in val_data_loader:
			noisy, net_input, mask = noisy.to(device).float(), net_input.to(device), mask.to(device)

			net_output = net(net_input)

			tmp_val_loss = loss_function(net_output*mask, noisy*mask).item() * correct_loss(mask)
			val_loss += tmp_val_loss
			n_losses += 1

			writer.add_scalar('Loss/val', tmp_val_loss, global_step=e)

		del noisy, net_input, mask, net_output, tmp_val_loss

		val_loss /= n_losses
		scheduler.step(val_loss)

		for noisy, net_input, mask in val_data_loader:
			noisy, net_input, mask = noisy.to(device).float(), net_input.to(device), mask.to(device)

			net_output = net(noisy)

			loss = loss_function(net_output, noisy).item()

			writer.add_scalar('Loss/whole_val', loss, global_step=e)

		del noisy, net_input, mask, net_output, loss

		print('\nTrain Loss (', e, ' \t', round(train_loss,6), 'val-loss\t', round(val_loss, 6))
		with open('loss.txt', 'a') as f:
			print(e, ';{:.10f}'.format(train_loss), ';{:.10f}'.format(val_loss), file=f)

		plot_denoise(net, test_data_loader, device, e, channels, valdir)

	return logdir


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

