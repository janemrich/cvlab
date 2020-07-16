import sys
sys.path.append('..')
from data import N2NDataset

import matplotlib.pyplot as plt

from torchvision import transforms
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from mask import Masker
from model import UNet

# TODO data set
n2s_sharp_train = N2NDataset()

model = UNet()

loss_function = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

dataloader = DataLoader(n2s_sharp_train, batch_size=32, shuffle=True)


# training
for i, batch in enumerate(dataloader):
    noisy_images, clean_images = batch

    net_input, mask = masker.mask(noisy_images, i)
    net_output = model(net_input)

    loss = loss_function(net_output*mask, noisy_images*mask)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if i % 10 == 0:
        print('Loss (', i, ' \t', round(loss.item(), 4))

    if i == 100:
        break

test_data_loader = DataLoader(n2s_sharp_test,
                                batch_size=32,
                                shuffle=False,
                                num_workers=3)
i, test_batch = next(enumerate(test_data_loader))
noisy, clean = test_batch
