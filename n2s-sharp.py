import sys
sys.path.append('..')
from data import N2SDataset
import model

import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from noise2self.mask import Masker
import model
import progress


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
            bar.inc_progress(i)

            if i % 10 == 0:
                print('Loss (', i, ' \t', round(loss.item(), 4))

        test_data_loader = DataLoader(test_dataset,
                                        batch_size=32,
                                        shuffle=False,
                                    num_workers=3)

        net.eval()
        i, test_batch = next(enumerate(test_data_loader))
        noisy = test_batch.float()
        plt.subplot(1,2,1)
        plt.imshow(noisy[0][0])
        plt.subplot(1,2,2)
        plt.imshow(net(noisy[:,1:,::]).detach()[0][0])
        plt.show()


# datasets
dataset = N2SDataset('data/sharp', target_size=(128, 128))

net = torch.nn.Sequential(
    model.ResBlock(1, 3, hidden_channels=[32, 32, 32]),
    #model.ResBlock(2, 3, hidden_channels=[16, 32, 16]),
model.ResBlock(1, 3, hidden_channels=[32, 32, 32])
)
net = net.float()

loss = MSELoss()

fit(net, loss, dataset, 30, batch_size=8)