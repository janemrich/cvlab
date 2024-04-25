import torch
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import model
from data import ProColoringDataset
from argparse import ArgumentParser
import json
from train import fit
import utils
import shutil
from eval import evaluate_color

def cli():
	parser = ArgumentParser()
	parser.add_argument("data", type=str, help="Root folder of the pro data")
	# parser.add_argument("--config", type=str, nargs="?", default="demosaic-default.json", help="path to configuration file for training setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	

if __name__=="__main__":
	args = cli()

	if args.name is None:
		args.name = config.get("name", None)

	# with open(args.config, 'r') as f:
		# config = json.load(f)

	# model_name = config.get("model", "unet")
	# model_params = config.get("model_params", {})
	
	# if  model_name == "resnet":
	# 	net = model.ResNet(2, 2, **model_params)
	# elif model_name == "unet":
	# 	net = model.UNet(2, **model_params)

	class HueLoss(torch.nn.Module):
		def __init__(self):
			super(HueLoss, self).__init__()
			self.mse = torch.nn.MSELoss()		

		def forward(self, y, _y):
			return self.mse(y[:, 0, :, :], _y[:, 0, :, :])

	net = torch.nn.Sequential(
		torch.nn.Conv2d(2, 50, kernel_size=1, bias=False),
		torch.nn.BatchNorm2d(100),
		torch.nn.LeakyReLU(),
		torch.nn.Conv2d(100, 100, kernel_size=1, bias=False),
		torch.nn.BatchNorm2d(100),
		# torch.nn.LeakyReLU(),
		# torch.nn.Conv2d(100, 100, kernel_size=1),
		# torch.nn.LeakyReLU(),
		# torch.nn.MaxPool2d(1, stride=1),
		# torch.nn.Conv2d(100, 100, kernel_size=1),
		# torch.nn.BatchNorm2d(100),
		# torch.nn.LeakyReLU(),
		# torch.nn.Conv2d(100, 100, kernel_size=1),
		torch.nn.LeakyReLU(),
		torch.nn.Conv2d(100, 100, kernel_size=1, bias=False),
		torch.nn.BatchNorm2d(100),
		torch.nn.LeakyReLU(),
		torch.nn.Conv2d(100, 100, kernel_size=1, bias=False),
		torch.nn.BatchNorm2d(100),
		torch.nn.LeakyReLU(),
		torch.nn.Conv2d(100, 3, kernel_size=1)
	)

	loss = torch.nn.MSELoss()
	
	dataset = ProColoringDataset(args.data, (128, 128))

	test_frac = 0.1
	train_dataset, test_dataset = utils.torch_random_split_frac(dataset, [1.0-test_frac, test_frac])

	resdir = fit(net, loss, train_dataset, device=args.device, name=args.name, batch_size=16, epochs=10, eval_fn=evaluate_color)