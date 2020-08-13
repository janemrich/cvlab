from torch.utils.data import random_split
from data import ProDemosaicDataset
import matplotlib.pyplot as plt
import numpy as np
import os

def torch_random_split_frac(dataset, fracs):
	assert sum(fracs)==1
	n = len(dataset)
	return random_split(dataset, [round(n*f) for f in fracs])

def to_rgb(hl):
	raise NotImplementedError

def evaluate_smithdata(dataset, model, resdir, device, step, n_images=2):
	# cast to ProDemosaicDataset
	while not isinstance(dataset, ProDemosaicDataset):
		dataset = dataset.dataset
	for i in range(n_images):
		X, Y, RGB = dataset.get_full(i)
		X, Y = X.unsqueeze(0).to(device), Y.unsqueeze(0).detach().numpy()

		model.to(device)
		model.eval()

		Y_ = model(X).detach().numpy()
		
		comp = np.concatenate([X.detach().numpy(), Y_, Y], axis=-1)
	
		plt.Figure(figsize=(10*n_images, 20))
		plt.tight_layout(h_pad=0, w_pad=0)
		
		sp = plt.subplot(n_images, 2, i*2+1)
		sp.axes.get_xaxis().set_visible(False)
		sp.axes.get_yaxis().set_visible(False)
		plt.title("{} high".format(i))
		plt.pcolormesh(comp[0, 0], vmin=0.0, vmax=1.0, cmap="Greys")
		
		sp = plt.subplot(n_images, 2, i*2+2)
		sp.axes.get_xaxis().set_visible(False)
		sp.axes.get_yaxis().set_visible(False)
		plt.title("{} low".format(i))
		plt.pcolormesh(comp[0, 1], vmin=0.0, vmax=1.0, cmap="Greys")
		
	plt.savefig(os.path.join(resdir, "eval{}.png".format(step)), dpi=500)