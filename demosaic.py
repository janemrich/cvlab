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
	
	return parser.parse_args()	

if __name__=="__main__":
	args = cli()
	
	with open(args.config, 'r') as f:
		config = json.load(f)

	# dataset = N2NDataset('openimages/train', target_size=(400, 400))
	# dataset = DemosaicingDataset('openimages/train', target_size=(400, 400))
	dataset = ProDemosaicDataset(args.data, **config.get("dataset", {}))

	net = torch.nn.Sequential(
		model.ResBlock(2, 3, hidden_channels=[32, 32, 32]),
		#model.ResBlock(2, 3, hidden_channels=[16, 32, 16]),
		model.ResBlock(2, 3, hidden_channels=[32, 32, 32])
	)

	loss = torch.nn.MSELoss()

	train_dataset, test_dataset = utils.torch_random_split_frac(dataset, [0.8, 0.2])

	fit(net, loss, train_dataset, device=args.device, **config.get("fit", {}))