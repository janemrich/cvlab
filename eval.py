from data import ProDemosaicDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os

def evaluate_smithdata(dataset, model, resdir, device, step, n_images=4):
	# cast to ProDemosaicDataset
	while not isinstance(dataset, ProDemosaicDataset):
		dataset = dataset.dataset
	# create DataLoader
	loader = DataLoader(dataset, batch_size=n_images, shuffle=False, num_workers=3)

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
		
	plt.savefig(os.path.join(resdir, "eval{}.png".format(step)), dpi=500)

def plot_denoise(net, data_loader, device, e, channels):
	i, test_batch = next(enumerate(data_loader))
	noisy = test_batch.to(device)
	noisy = noisy.float()
	denoised = net(noisy[:,:channels,::]).detach()
	noisy = noisy[:,:channels,::]
	noisy, denoised = noisy.cpu(), denoised.cpu()
	comp = np.concatenate([noisy, denoised], axis=-2)
	if channels == 2:
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

