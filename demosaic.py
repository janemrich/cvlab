import torch
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import model
from data import ProDemosaicDataset
from argparse import ArgumentParser
import json
from train_demosaic import fit
import utils
import shutil
from noise2self.models.unet import UNet as N2SUnet

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
	parser.add_argument("--config", type=str, nargs="?", default="demosaic-default.json", help="path to configuration file for training setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	

if __name__=="__main__":
	args = cli()

	if args.name is None:
		args.name = config.get("name", None)

	with open(args.config, 'r') as f:
		config = json.load(f)

	# dataset = N2NDataset('openimages/train', target_size=(400, 400))
	# dataset = DemosaicingDataset('openimages/train', target_size=(400, 400))

	model_name = config.get("model", "unet")
	model_params = config.get("model_params", {})
	
	if  model_name == "resnet":
		net = model.ResNet(2, 2, **model_params)
	elif model_name == "unet":
		net = model.UNet(2, **model_params)
	elif model_name == "n2s-unet":
		from noise2self.models.unet import Unet
		net = Unet(n_channel_in=2, n_channel_out=2, **model_params)

	loss_fun = config.get("loss", "mse")
	if loss_fun == 'mse':
		loss = torch.nn.MSELoss()
	elif loss_fun == 'smoothmse':	
		loss = model.SmoothMSELoss(2, 0.1)
	
	dataset = ProDemosaicDataset(args.data, **config.get("dataset", {}))

	pretrain = config.get("pretrain", None)
	if pretrain is not None:
		pretrain_dataset = ProDemosaicDataset(pretrain["data"], **pretrain.get("dataset", {}))
		fit(net, loss, pretrain_dataset, device=args.device, name=args.name, **pretrain.get("fit", {}), pretrain=True)

	test_frac = config.get("test_frac", 0.1)
	train_dataset, test_dataset = utils.torch_random_split_frac(dataset, [1.0-test_frac, test_frac])

	resdir = fit(net, loss, train_dataset, device=args.device, name=args.name, **config.get("fit", {}))

	shutil.copyfile(args.config, os.path.join(resdir, os.path.basename(args.config)))
	
	# torch.save(net, os.path.join(resdir, "model.sav"))
	torch.save(net.state_dict(), os.path.join(resdir, "statedict.pt"))