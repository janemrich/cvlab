from data import RawDataset

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""
script for plotting histograms over smith data
"""

dataset = RawDataset('data/sharp', sharp=True, complete_background_noise=True)

maxval = 65535 + 5000
bin_minval = -100
bins = maxval + abs(bin_minval)

histogram = np.histogram(dataset.__getitem__(0), bins=bins, range=(bin_minval, maxval))[0]
for i, image in enumerate(dataset):
    print(i, flush=True, end='\r')
    curr_histogram = np.histogram(image, bins=bins, range=(bin_minval, maxval))
    histogram += curr_histogram[0]
    # break


def plot_histogram_values(histogram, title, bins, minval=0):
    ticks = 20
    labels = range(minval, maxval, int((maxval-minval)/ticks))
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Histogram of sharp machine intensities')
    plt.plot(range(bin_minval, maxval), histogram)
    plt.xticks(range(0, bins, int(bins/ticks)), labels)

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plot_gauss()
    plot_fake_gauss()


    plt.show()

def plot_gauss():
    from scipy.stats import norm
    rv = norm(loc=65535, scale=1600)
    x = range(0, bins)
    plt.plot(x, 26700 * rv.pdf(x))

def plot_fake_gauss():
    from scipy.stats import poisson
    rv = poisson(2684760 / 0.02441)
    x = range(maxval-5000, maxval)
    plt.plot(x, 100000 * 26700 * rv.pmf(x))

def plot_possion():
    from scipy.stats import poisson
    rv = poisson(2684760)
    x = range(2500000, 2800000)
    plt.plot(x, rv.pmf(x))
    plt.show()

log_histogram = np.log(histogram)

plot_histogram_values(histogram, 'raw', bins)
plot_histogram_values(log_histogram, 'log', bins)
#plot_histogram_values(log_histogram[-512:], 'log', bins=512, minval=int(maxval-(512*maxval/bins)))
plot_possion()