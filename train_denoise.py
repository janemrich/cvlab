import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from noise2self.mask import Masker

import model
import progress
from eval import plot_denoise

def fit(net, loss_function, dataset, epochs, target_size, batch_size=32, device='cpu', mask_grid_size=4, fade_threshold=0, channels=2, **kwargs):

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

	test_size = int(0.5 * len(test_dataset))
	val_size = len(test_dataset) - test_size

	test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, val_size], generator=torch.Generator().manual_seed(42))

	dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
	val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

	masker = Masker(width = mask_grid_size, mode='interpolate')

	optimizer_args_default = {'lr': 0.001}
	optimizer_args = {k[len("optimizer_"):]: v for k, v in kwargs.items() if k.startswith("optimizer_")}
	
	scheduler_args_default = {"patience":1, "cooldown":4}
	scheduler_args = {k[len("scheduler_"):]: v for k, v in kwargs.items() if k.startswith("scheduler_")}
	
	optimizer = torch.optim.Adam(net.parameters(), **{**optimizer_args_default, **optimizer_args})
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{**scheduler_args_default, **scheduler_args})

	net.to(device)
	loss_function.to(device)

	plot_denoise(net, test_data_loader, device, 0, channels)

	# training
	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=train_size)
		net.train()
		train_loss = 0.0
		n_losses = 0
		rng = np.random.default_rng()

		# allocate memor
		normal_batch = next(iter(dataloader))[:,:channels,::].float()
		zero_mask = torch.zeros(normal_batch.shape[-2:]).to(device)
		mask_low = torch.empty(normal_batch.shape[-2:])
		mask_high = torch.empty(normal_batch.shape[-2:])
		if channels == 1:
			mask = torch.empty(normal_batch.shape[-3:])
		else:
			mask = torch.empty(normal_batch.shape[-2:])

		for batch in dataloader:
			i = rng.integers(mask_grid_size**2)
			loss_channel = rng.integers(2)
			noisy_images = batch.to(device)
			noisy_images = noisy_images.float()

			noisy_images = noisy_images[:,:channels,::]
			if channels == 1:
				net_input, mask = masker.mask(noisy_images, i)
			elif channels == 2:
				net_input = torch.empty_like(noisy_images)
				if loss_channel == 1:
					net_input[:,:1,:,:], mask_high = masker.mask(noisy_images[:,:1,:,:], i, pair=True, pair_direction='right')
					net_input[:,1:,:,:], mask_low = masker.mask(noisy_images[:,1:,:,:], i)
					# only take loss of masking where only one pixel is masked
					mask = torch.stack([zero_mask, mask_low], axis=-3)
				else:
					net_input[:,:1,:,:], mask_high = masker.mask(noisy_images[:,:1,:,:], i) # because high channel is shifted to the left
					net_input[:,1:,:,:], mask_low = masker.mask(noisy_images[:,1:,:,:], i, pair=True, pair_direction='left')
					mask = torch.stack([mask_high, zero_mask], axis=-3)
			else:
				return NotImplementedError

			net_output = net(net_input)

			fade = transforms.Lambda(lambda x: fading_loss(x, threshold_from_end=fade_threshold))
			fade_factor = fade(noisy_images*mask)

			loss = loss_function(net_output*mask, noisy_images*mask*fade_factor)
			train_loss += loss.item()
			n_losses += 1

			optimizer.zero_grad()

			loss.backward()

			optimizer.step()
			bar.inc_progress(batch_size)
		train_loss /= n_losses
		del batch, noisy_images, net_input, net_output, loss 

		net.eval()
		val_loss = 0.0
		n_losses = 0
		for i, batch in enumerate(val_data_loader):
			noisy = batch.to(device)
			noisy = noisy.float()

			noisy = noisy[:,:channels,::]
			net_input, mask = masker.mask(noisy, i)
			net_output = net(net_input)

			val_loss += loss_function(net_output*mask, noisy*mask).item()
			n_losses += 1
		del batch, noisy, net_input, net_output

		val_loss /= n_losses
		scheduler.step(val_loss)
		loss_display_factor = target_size[0] * target_size[1]
		print('\nTrain Loss (', e, ' \t', round(train_loss,6), 'val-loss\t', round(val_loss,6))
		with open('loss.txt', 'a') as f:
			print(e, ';{:.10f}'.format(train_loss), ';{:.10f}'.format(val_loss), file=f)

		plot_denoise(net, test_data_loader, device, e, channels)

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

