import torch
import torchvision.transforms
from PIL import Image, ImageOps
from torch.utils.tensorboard import SummaryWriter
import progress
import utils
from eval import evaluate_smithdata
import os
from datetime import datetime

def fit(net, criterion, dataset, epochs=3, batch_size=24, device="cpu", name=None, pretrain=False, val_frac=0.1, **kwargs):
	if "cuda" in device:
		torch.backends.cudnn.benchmark = True
	if pretrain:
		name = name + "-pre"
	logdir = os.path.join('runs', name + datetime.now().strftime("_%d%b-%H%M%S")) if name is not None else None
	writer = SummaryWriter(log_dir=logdir)

	ds_train, ds_val = utils.torch_random_split_frac(dataset, [1.0-val_frac, val_frac])

	loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
	loader_test = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)
	
	net.to(device)
	criterion.to(device)

	optimizer_args_default = {'lr': 0.001}
	optimizer_args = {k[len("optimizer_"):]: v for k, v in kwargs.items() if k.startswith("optimizer_")}
	
	scheduler_args_default = {"patience":1, "cooldown":4, "factor": 0.5}
	scheduler_args = {k[len("scheduler_"):]: v for k, v in kwargs.items() if k.startswith("scheduler_")}
	
	optimizer = torch.optim.Adam(net.parameters(), **{**optimizer_args_default, **optimizer_args})
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{**scheduler_args_default, **scheduler_args})

	valdir = os.path.join(writer.log_dir, "val")
	os.mkdir(valdir)
	
	n = 4

	global_step = 0
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
			writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
			bar.inc_progress(len(X))
			global_step += 1
		del X, Y, Y_, loss

		bar = progress.Bar("Epoch {}, test".format(e), finish=len(ds_val))
		net.eval()
		losses = 0
		n_losses = 0
		first = True
		with torch.no_grad():	
			for X, Y in loader_test:
				X, Y = X.to(device), Y.to(device)
				Y_ = net(X)
				# if first:
				# 	first = False
				# 	vis = torch.cat([X[:n], Y_[:n], Y[:n]], dim=-1)
				# 	writer.add_images("validation_prediction_high", vis[:, 0:1], global_step=e)
				# 	writer.add_images("validation_prediction_low", vis[:n, 1:2], global_step=e)
				losses = losses + criterion(Y, Y_).item()
				n_losses += 1
				bar.inc_progress(len(X))
			loss = losses / n_losses
			writer.add_scalar('Loss/val', losses / n_losses, global_step=e)
			# scheduler.step(loss)
			
			evaluate_smithdata(dataset, net, valdir, device, e)

			del X, Y, Y_, loss, losses

		# try:	
		# 	dataset.reset()
		# except:
		# 	try:
		# 		dataset.dataset.reset()
		# 	except:
		# 		dataset.dataset.datatset.reset()

	return logdir
