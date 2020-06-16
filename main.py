import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import model
from data import N2NDataset, DemosaicingDataset
import progress


device = 'cuda'

def fit(net, criterion, dataset, epochs, batch_size=24):
	writer = SummaryWriter()

	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.3)

	for e in range(epochs):
		bar = progress.Bar("Epoch {}".format(e), finish=len(dataset))
		net.train()
		
		for X, Y in loader:
			X, Y = X.to(device), Y.to(device)
			optimizer.zero_grad()
			Y_ = net(X)
			loss = criterion(Y, Y_)
			loss.backward()
			optimizer.step()
			writer.add_scalar('Loss/train', loss.item())
			bar.inc_progress(batch_size)
		scheduler.step()

		net.eval()
		plt.figure(figsize=(15, 15))
		for i in range(0,6,2):
			plt.subplot(3, 2, i+1)
			plt.imshow(dataset[i][0].permute(1, 2, 0))
			plt.subplot(3, 2, i+2)
			plt.imshow(net(torch.unsqueeze(dataset[i][0], 0).to(device)).to('cpu').squeeze().permute(1, 2, 0).detach())
		plt.savefig('epoch{}predicions.jpg'.format(e))

if __name__=="__main__":
	# dataset = N2NDataset('openimages/train', target_size=(400, 400))
	dataset = DemosaicN2NDataset('openimages/train', target_size=(400, 400))

	net = torch.nn.Sequential(
		model.ResBlock(3, 3, hidden_channels=[16, 32, 48, 64]),
		model.ResBlock(3, 3, hidden_channels=[16, 32, 48, 64]),
		model.ResBlock(3, 3, hidden_channels=[16, 32, 48, 64])
	)

	loss = torch.nn.MSELoss()

	fit(net, loss, dataset, 4, batch_size=8)

	

