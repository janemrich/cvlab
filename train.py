import torch
import torchvision.transforms
from PIL import Image, ImageOps
from torch.utils.tensorboard import SummaryWriter
import progress
import utils
import os
from datetime import datetime

def fit(net, criterion, dataset, epochs=3, batch_size=24, device="cpu", name=None):
	logdir = os.path.join('runs', name + datetime.now().strftime("_%d%b-%H%M%S")) if name is not None else None
	writer = SummaryWriter(log_dir=logdir)

	ds_train, ds_val = utils.torch_random_split_frac(dataset, [0.9, 0.1])


	loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
	loader_test = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)

	first_batch = next(iter(loader_test))
	n = max(len(first_batch), 8)
	writer.add_images("validation_in_high", first_batch[0][:n, 0:1])
	writer.add_images("validation_gt_high", first_batch[1][:n, 0:1])
	
	net.to(device)

	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, cooldown=2)

	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=len(ds_train))
		net.train()
		first = False
		for X, Y in loader_train:
			X, Y = X.to(device), Y.to(device)
			optimizer.zero_grad()
			Y_ = net(X)
			loss = criterion(Y, Y_)
			loss.backward()
			optimizer.step()
			writer.add_scalar('Loss/train', loss.item())
			if not first:
				first=True
				writer.add_images("train_prediction_high", Y_[:n, 0:1], global_step=e)
				writer.add_images("train_prediction_low", Y_[:n, 1:2], global_step=e)
			
			bar.inc_progress(len(X))

		bar = progress.Bar("Epoch {}, test".format(e), finish=len(ds_val))
		net.eval()
		losses = 0
		n_losses = 0
		first = True
		
		for X, Y in loader_test:
			X, Y = X.to(device), Y.to(device)
			Y_ = net(X)
			if first:
				first = False
				writer.add_images("validation_prediction_high", Y_[:n, 0:1], global_step=e)
				writer.add_images("validation_prediction_low", Y_[:n, 1:2], global_step=e)
			losses = losses + criterion(Y, Y_).item()
			n_losses += 1
			bar.inc_progress(len(X))
		loss = losses / n_losses
		writer.add_scalar('Loss/val', losses / n_losses)
		scheduler.step(loss)
