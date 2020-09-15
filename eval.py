from data import ProDemosaicDataset, ProColoringDataset, N2SProDemosaicDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os
from matplotlib.colors import hsv_to_rgb

def evaluate_color(dataset, model, resdir, device, step, n_images=4):
	# cast to ProDemosaicDataset
	while not isinstance(dataset, ProColoringDataset):
		dataset = dataset.dataset
	# create DataLoader
	loader = DataLoader(dataset, batch_size=n_images, shuffle=False, num_workers=4)

	X, Y = next(iter(loader))
	X, Y = X.to(device), Y.cpu().detach().numpy()

	model.to(device)
	model.eval()

	Y_ = model(X).cpu().detach().numpy()
	
	comp = np.concatenate([Y_, Y], axis=-2)

	fig = plt.figure(figsize=(3*n_images, 8))
	fig.suptitle('top: IN, middle: OUT, bottom: GT', fontsize=16)
	for j in range(n_images):
		ax = fig.add_subplot(2,n_images,j+1)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.imshow(X[j][0], cmap="gray", vmin=0.0, vmax=1.0)
		ax1 = fig.add_subplot(2,n_images,n_images+j+1)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title("image {}".format(j))
		ax1.imshow(hsv_to_rgb(np.transpose(comp[j], (1, 2, 0))), vmin=0.0, vmax=1.0)
		
	plt.savefig(os.path.join(resdir, "eval{}.png".format(step)), dpi=400)


def evaluate_smithdata(dataset, model, resdir, device, step, n_images=4):
	# cast to ProDemosaicDataset
	while not isinstance(dataset, ProDemosaicDataset):
		dataset = dataset.dataset
	# create DataLoader
	loader = DataLoader(dataset, batch_size=n_images, shuffle=False, num_workers=4)

	X, Y = next(iter(loader))
	X, Y = X.to(device), Y.cpu().detach().numpy()

	model.to(device)
	model.eval()

	Y_ = model(X).cpu().detach().numpy()
	
	comp = np.concatenate([X.cpu().detach().numpy(), Y_, Y], axis=-2)

	fig = plt.figure(figsize=(3*n_images, 8))
	fig.suptitle('top: IN, middle: OUT, bottom: GT', fontsize=16)
	for j in range(n_images):
		ax1 = fig.add_subplot(1,n_images,j+1)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title("image {}".format(j))
		ax1.imshow(comp[j][0], interpolation=None, vmin=0.0, vmax=1.0, cmap='gray')
		
	plt.savefig(os.path.join(resdir, "eval{}.png".format(step)), dpi=400)

def evaluate_joint(dataset, model, resdir, device, step, n_images=4):
	dataset.deterministic = True
	# create DataLoader
	loader = DataLoader(dataset, batch_size=n_images, shuffle=False, num_workers=1)

	model.eval()
	noisy, _, mask, sharp = next(iter(loader))
	sharp = sharp.to(device)
	sharp = sharp.float()
	denoised = model(sharp).detach()
	sharp, denoised = sharp.cpu(), denoised.cpu()
	comp = np.concatenate([sharp, denoised, noisy], axis=-2)
	comp = np.concatenate([comp[:,:1,:,:], comp[:,1:,:,:]], axis=-1)

	fig = plt.figure(figsize=(6*n_images, 11))
	fig.suptitle('top: IN, middle: OUT, bottom: GT', fontsize=16)
	for j in range(n_images):
		ax1 = fig.add_subplot(1,n_images,j+1)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title("image {}".format(j))
		ax1.imshow(comp[j][0], interpolation=None, vmin=0.0, vmax=1.0, cmap='gray')
		
	plt.savefig(os.path.join(resdir, "eval{}.png".format(step)), dpi=400)
	# plt.close()
	del loader, sharp, noisy, mask, denoised, comp, _

def plot_denoise(net, data_loader, device, e, channels):
	noisy, net_input, mask = next(iter(data_loader))
	noisy = noisy.to(device)
	noisy = noisy.float()
	# denoised = net(noisy[:,:channels,::]).detach()
	denoised = net(noisy).detach()
	noisy = noisy[:,:channels,::]
	noisy, denoised = noisy.cpu(), denoised.cpu()
	comp = np.concatenate([noisy, denoised], axis=-2)
	if channels == 2:
		# [low, high]
		comp = np.concatenate([comp[:,:1,:,:], comp[:,1:,:,:]], axis=-1)

	n_pics = 3
	fig = plt.figure(figsize=(6*n_pics, 7))
	#fig.suptitle('channel low, noisy(top) vs denoised(bottom)', fontsize=30)
	for j in range(n_pics):
		# define images to show
		if j == 0:
			k = 3
		if j == 2:
			k = 4
		else:
			k = j
		ax1 = fig.add_subplot(1,n_pics,j+1)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title("image {}".format(j))
		ax1.imshow(comp[j][0], interpolation=None, vmin=0.0, vmax=1.0, cmap='gray')

	plt.savefig('n2s_epoch' + str(e) + '.png', dpi=300)
	del noisy, denoised, comp

def plot_denoising_masking(noisy, net_input, mask, net_output):
	fig = plt.figure()
	titles = ['noisy low', 'net input low', 'mask low', 'net input - noisy', 'net output', 'noisy high', 'net input high', 'mask high', 'net input - noisy', 'net output']
	images = [noisy[0,0], net_input[0,0], mask[0,0], net_input[0,0] - noisy[0,0], net_output.detach()[0,0], noisy[0,1], net_input[0,1], mask[0,1], net_input[0,1] - noisy[0,1], net_output.detach()[0,1]]

	for i, (title, im) in enumerate(zip(titles, images)):
		ax = fig.add_subplot(2,5,i+1)
		ax.set_title(title)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.imshow(im, interpolation=None, vmin=0.0, vmax=1.0, cmap='gray')
	plt.show()

def plot_sharp_masking(patch, patch_low, patch_high, sharp_sparse, sharp):
	fig = plt.figure()
	titles = ['orig patch low', 'patch low', 'sparse low', 'sparse low', 'orig patch high', 'patch high', 'sparse high', 'sharp high']
	tensors = [patch[0], patch_low, sharp_sparse[0], sharp[0], patch[1], patch_high,  sharp_sparse[1], sharp[1]]

	for i, (title, im) in enumerate(zip(titles, tensors)):
		ax = fig.add_subplot(2,4,i+1)
		ax.set_title(title)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.imshow(im, interpolation=None, vmin=0.0, vmax=1.0, cmap='inferno')
	plt.show()

def plot_tensors(tensors, v=False):
	fig = plt.figure()

	for i, im in enumerate(tensors):
		ax = fig.add_subplot(tensors.__len__(), tensors[0].shape[-3], 2*i+1)
		ax.get_xaxis().set_visible(v)
		ax.get_yaxis().set_visible(v)
		plt.imshow(im[0], interpolation=None, vmin=0.0, vmax=1.0, cmap='inferno')

		ax = fig.add_subplot(tensors.__len__(), tensors[0].shape[-3], 2*i+2)
		ax.get_xaxis().set_visible(v)
		ax.get_yaxis().set_visible(v)
		plt.imshow(im[1], interpolation=None, vmin=0.0, vmax=1.0, cmap='inferno')
	plt.show()
