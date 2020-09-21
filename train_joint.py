import torch
import torchvision.transforms
from PIL import Image, ImageOps
from torch.utils.tensorboard import SummaryWriter
import progress
import utils
from utils import correct_loss
from eval import evaluate_joint, evaluate_color
import os
from datetime import datetime

def fit(net, criterion, dataset, epochs=3, batch_size=24, device="cpu", name=None, pretrain=False, val_frac=0.1, mask_grid_size=6, eval_fn=evaluate_joint, **kwargs):
	if "cuda" in device:
		torch.backends.cudnn.benchmark = True
	if pretrain:
		name = name + "-pre"
	logdir = os.path.join('runs', name + datetime.now().strftime("_%d%b-%H%M%S")) if name is not None else None
	writer = SummaryWriter(log_dir=logdir)

	train_size = int((1.0-val_frac) * len(dataset))
	test_size = len(dataset) - train_size
	ds_train, ds_val = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

	loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
	loader_test = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=3)
	
	net.to(device)
	criterion.to(device)

	learn_rate = kwargs.get('learn_rate', 0.002)
	optimizer_args_default = {'lr': learn_rate * (batch_size/32)}
	optimizer_args = {k[len("optimizer_"):]: v for k, v in kwargs.items() if k.startswith("optimizer_")}
	
	scheduler_args_default = {"patience":1, "cooldown":4, "factor": 0.5}
	scheduler_args = {k[len("scheduler_"):]: v for k, v in kwargs.items() if k.startswith("scheduler_")}
	
	optimizer = torch.optim.Adam(net.parameters(), **{**optimizer_args_default, **optimizer_args})
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{**scheduler_args_default, **scheduler_args})

	valdir = os.path.join(writer.log_dir, "val")
	os.mkdir(valdir)
	
	eval_fn(ds_val, net, valdir, device, 0)

	global_step = 0
	for e in range(epochs):
		bar = progress.Bar("Epoch {}, train".format(e), finish=len(ds_train))
		net.train()
		for noisy, net_input, mask, _ in loader_train:
			noisy, net_input, mask = noisy.to(device), net_input.to(device), mask.to(device)
			optimizer.zero_grad()
			net_output = net(net_input)
			loss = criterion(net_output*mask, noisy*mask)
			loss = loss * correct_loss(mask)
			loss.backward()
			optimizer.step()
			writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
			bar.inc_progress(batch_size)
			global_step += 1
		del noisy, net_input, net_output, mask, loss, _

		bar = progress.Bar("Epoch {}, test".format(e), finish=len(ds_val))
		net.eval()
		losses = 0
		n_losses = 0

		with torch.no_grad():	
			for noisy, net_input, mask, _ in loader_test:
				noisy, net_input, mask = noisy.to(device).float(), net_input.to(device), mask.to(device)

				net_output = net(net_input)
				losses += criterion(net_output*mask, noisy*mask).item() * correct_loss(mask)
				n_losses += 1
				bar.inc_progress(len(noisy))
			loss = losses / n_losses
			writer.add_scalar('Loss/val', losses / n_losses, global_step=e)
			scheduler.step(loss)
			
			eval_fn(ds_val, net, valdir, device, e)

			del noisy, net_input, net_output, mask, loss, losses, _

	return logdir
