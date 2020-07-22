import torch
import torchvision.transforms
from PIL import Image, ImageOps
from torch.utils.tensorboard import SummaryWriter
import progress
import utils

def fit(net, criterion, dataset, epochs=3, batch_size=24, device="cpu"):
	writer = SummaryWriter()

	ds_train, ds_val = utils.torch_random_split_frac(dataset, [0.9, 0.1])


	loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
	loader_test = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)

	first_batch = next(iter(loader_test))
	writer.add_images("validation_in_high", first_batch[0][:, 0:1])
	writer.add_images("validation_in_high", first_batch[0][:, 1:2])
	writer.add_images("validation_gt_high", first_batch[1][:, 0:1])
	writer.add_images("validation_gt_high", first_batch[1][:, 1:2])
	
	net.to(device)

	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.3)

	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=len(ds_train))
		net.train()
		for X, Y in loader_train:
			X, Y = X.to(device), Y.to(device)
			optimizer.zero_grad()
			Y_ = net(X)
			loss = criterion(Y, Y_)
			loss.backward()
			optimizer.step()
			writer.add_scalar('Loss/train', loss.item())
			bar.inc_progress(len(X))
		scheduler.step()

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
				writer.add_images("validation_prediction_high", Y_[:, 0:1], global_step=e)
				writer.add_images("validation_prediction_low", Y_[:, 1:2], global_step=e)
			losses = losses + criterion(Y, Y_).item()
			n_losses += 1
			bar.inc_progress(len(X))
		writer.add_scalar('Loss/val', losses / n_losses)
