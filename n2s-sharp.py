import sys
sys.path.append('..')
from data import N2SDataset
import model

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json

import torch
from torchvision import transforms
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from noise2self.mask import Masker
import model
import progress

"""
Config format:
{
	model: "res"/",
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
	parser.add_argument("--config", type=str, nargs="?", default="denoising-default.json", help="path to configuration file for training setup")
	parser.add_argument("--device", type=str, nargs="?", default="cpu")
	parser.add_argument("--name", type=str, nargs="?", default=None)
	
	return parser.parse_args()	


def fit(net, loss_function, dataset, epochs, batch_size=32):

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    masker = Masker(width = 4, mode='interpolate')
    optimizer = Adam(net.parameters(), lr=0.001)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # training
    for e in range(epochs):
        bar = progress.Bar("Epoch {}, train".format(e), finish=train_size)
        net.train()
        for i, batch in enumerate(dataloader):
            noisy_images = batch
            noisy_images = noisy_images.float()

            # for now only low input
            noisy_images = noisy_images[:,:1,::]
            net_input, mask = masker.mask(noisy_images, i)
            net_output = net(net_input)

            loss = loss_function(net_output*mask, noisy_images*mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            bar.inc_progress(batch_size)

            # if i % 10 == 0:
        
        print('\nLoss (', i, ' \t', round(loss.item(), 6))
   
        test_data_loader = DataLoader(test_dataset,
                                        batch_size=32,
                                        shuffle=False,
                                    num_workers=3)

        net.eval()
        plot_val(net, test_data_loader)

def plot_val(net, data_loader):
    i, test_batch = next(enumerate(data_loader))
    noisy = test_batch.float()
    denoised = net(noisy[:,1:,::]).detach()
    n_pics = 5
    for j in range(n_pics):
        fig = plt.subplot(2,n_pics,j+1)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.imshow(noisy[j][0], cmap='gray')
        fig = plt.subplot(2,n_pics,n_pics+j+1)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.imshow(denoised[j][0], cmap='gray')
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('n2s.png')


if __name__=="__main__":
	args = cli()
	
	with open(args.config, 'r') as f:
		config = json.load(f)

	# datasets
	dataset = N2SDataset(args.data, sharp=True, **config.get("dataset", {}))

	net_full_residual = torch.nn.Sequential(
		model.ConvBlock(1, 16, 3, padding_mode='reflect'),
		model.ResBlock(16, 3, padding_mode='reflect', hidden_channels=[32, 32, 32]),
		model.ResBlock(16, 3, padding_mode='reflect', hidden_channels=[32, 32, 32]),
		model.OutConv(16, 1) 
	)
	net_res = torch.nn.Sequential(
		# model.ConvBlock(1, 16, 3, padding_mode='reflect'),
		model.ResBlock(1, 3, padding_mode='reflect', activation='relu', hidden_channels=[32, 32, 32]),
		model.ResBlock(1, 3, padding_mode='reflect', activation='relu', hidden_channels=[32, 32, 32]),
		#model.ConvBlock(1, 1, 3, padding_mode='reflect', activation='sigmoid'),
	)

	#from noise2self.models.babyunet import BabyUnet
	#net_babyu = BabyUnet()


	if config['model'] == 'res':
		net = net_res
	net = net.float()

	loss = MSELoss()

	fit(net, loss, dataset, 15, batch_size=32)