import os
import sys
sys.path.append('..')

import numpy as np
from argparse import ArgumentParser
import json
import torch
from torch.nn import MSELoss

from data import N2SDataset
import model
from train_denoise import fit
import shutil

"""
Config format:
{
	model: "res"/",
	model_params: {},
	train: {
		batch_size: int,
		epochs: int
	}
}
"""

def cli():
	parser = ArgumentParser()
	parser.add_argument("data", type=str, help="Root folder of the pro data")
	parser.add_argument("--config", type=str, nargs="?", default="denoising-default.json", help="path to configuration file for training setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	

if __name__=="__main__":
	args = cli()
	
	with open(args.config, 'r') as f:
		config = json.load(f)

	if args.name is None:
		train = config.get("train")
		args.name = config.get("name") + config.get("model", None) + "_lr" + str(train.get("learn_rate")) + "_b" + str(train.get("batch_size")) + "_g" + str(train.get("mask_grid_size")) + "_hp" + str(config.get("dataset").get("halfpixel"))

	channels = config['channels']

	# datasets
	dataset = N2SDataset(args.data, mask_grid_size=config['train']['mask_grid_size'], channels=config['channels'], **config.get("dataset", {}))

	model_type = config['model']
	if model_type == 'unet':
		from model import UNet
		net = UNet(channels, **config.get('model_params', {}))
	if model_type == 'resnet':
		from model import ResNet
		net = ResNet(channels, channels, padding_mode='reflect', **config.get("model_params", {}))# in, out channels
	if model_type == 'n2s-babyu':
		from noise2self.models.babyunet import BabyUnet
		net = BabyUnet(channels, channels)
	if model_type == 'n2s-unet':
		from noise2self.models.unet import Unet
		net = Unet(n_channel_in=channels, n_channel_out=channels, **config.get("model_params", {}))
	if model_type == 'n2s-dncnn':
		from noise2self.models.dncnn import DnCNN
		net = DnCNN(channels) # number of channels

	net = net.float()

	loss = MSELoss()

	resdir = fit(net,
		loss,
		dataset,
		config['train']['epochs'],
		config['dataset']['target_size'],
		batch_size=config['train']['batch_size'],
		device=args.device,
		name=args.name,
		mask_grid_size=config['train']['mask_grid_size'],
		fade_threshold=config['train']['fade_threshold'],
		channels=channels,
		learn_rate=config['train']['learn_rate']
		)

	shutil.copyfile(args.config, os.path.join(resdir, os.path.basename(args.config)))

	torch.save(net.state_dict(), os.path.join(resdir, "statedict.pt"))