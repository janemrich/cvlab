from data import RawDataset

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""
script for plotting histograms over smith data
"""

dataset = RawDataset('data/sharp', sharp=True)

maxval = 65535
bins = 4096
print('binsize: ', maxval/bins)

histogram = np.histogram(dataset.__getitem__(0), bins=bins, range=(0.0, 65535.0))[0]
for i, image in enumerate(dataset):
    print(i, flush=True, end='\r')
    curr_histogram = np.histogram(image, bins=bins, range=(0.0, 65535.0))
    histogram += curr_histogram[0]


def plot_histogram_values(histogram, title, bins, minval=0):
    ticks = 10
    labels = range(minval, maxval, int((maxval-minval)/ticks))
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Histogram of sharp machine intensities')
    plt.plot(range(bins), histogram)
    plt.xticks(range(0, bins, int(bins/ticks)), labels)
    plt.show()


log_histogram = np.log(histogram)

plot_histogram_values(histogram, 'raw', bins)
plot_histogram_values(log_histogram, 'log', bins)
plot_histogram_values(log_histogram[-1000:], 'log', bins=1000, minval=int(maxval-(1000*maxval/bins)))