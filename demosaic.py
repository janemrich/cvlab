import torch
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import model
from data import N2NDataset, DemosaicingDataset, ProDemosaicDataset
from argparse import ArgumentParser
import json
from train import fit
import utils

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
	
	with open(args.config, 'r') as f:
		config = json.load(f)

	# dataset = N2NDataset('openimages/train', target_size=(400, 400))
	# dataset = DemosaicingDataset('openimages/train', target_size=(400, 400))
	dataset = ProDemosaicDataset(args.data, **config.get("dataset", {}))

	model_name = config.get("model", "unet")
	model_params = config.get("model_params", {})
	if  model_name == "resnet":
		net = model.ResNet(2, 2, **model_params)
	elif model_name == "unet":
		net = model.UNet(2, **model_params)

	loss = torch.nn.MSELoss()

	train_dataset, test_dataset = utils.torch_random_split_frac(dataset, [0.8, 0.2])

	fit(net, loss, train_dataset, device=args.device, name=args.name, **config.get("fit", {}))