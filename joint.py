import torch
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import model
from data import N2SProDemosaicDataset
from argparse import ArgumentParser
import json
from train_joint import fit
import utils
import shutil

"""
Config format:
{
	model: "unet"/"resnet"/"densenet",
	model_params: {},
	fit: {
		batch_size: int,
		epochs: int
	}
}
"""

def cli():
	parser = ArgumentParser()
	parser.add_argument("data", type=str, help="Root folder of the pro data")
	parser.add_argument("--config", type=str, nargs="?", default="joint-default.json", help="path to configuration file for fiting setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	

if __name__=="__main__":
	args = cli()

	with open(args.config, 'r') as f:
		config = json.load(f)

	if args.name is None:
		train = config.get("fit")
		args.name = config.get("name") + config.get("model", None) + "_lr" + str(train.get("learn_rate")) + "_b" + str(train.get("batch_size")) + "_g" + str(train.get("mask_grid_size")) + "_hp" + str(config.get("dataset").get("halfpixel"))

	model_name = config.get("model", "unet")
	model_params = config.get("model_params", {})
	
	if  model_name == "resnet":
		net = model.ResNet(2, 2, **model_params)
	elif model_name == "unet":
		net = model.UNet(2, **model_params)
	elif model_name == 'n2s-unet':
		from noise2self.models.unet import Unet
		net = Unet(n_channel_in=2, n_channel_out=2, **config.get("model_params", {}))

	loss_fun = config.get("loss", "mse")
	if loss_fun == 'mse':
		loss = torch.nn.MSELoss()
	elif loss_fun == 'smoothmse':	
		loss = model.SmoothMSELoss(2, 0.1)
	
	dataset = N2SProDemosaicDataset(args.data, mask_grid_size=config['fit']['mask_grid_size'], **config.get("dataset", {}))

	prefit = config.get("pretrain", None)
	if prefit is not None:
		prefit_dataset = N2SProDemosaicDataset(pretrain["data"], **pretrain.get("dataset", {}))
		fit(net, loss, prefit_dataset, device=args.device, name=args.name, **pretrain.get("fit", {}), pretrain=True)

	test_frac = config.get("test_frac", 0.1)
	train_size = int((1.0-test_frac) * len(dataset))
	test_size = len(dataset) - train_size
	fit_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

	resdir = fit(net, loss, fit_dataset, device=args.device, name=args.name, **config.get("fit", {}))

	shutil.copyfile(args.config, os.path.join(resdir, os.path.basename(args.config)))

	torch.save(net.state_dict(), os.path.join(resdir, "statedict.pt"))
