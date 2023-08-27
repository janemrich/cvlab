# Deep Unsupervised Denoising of X-Ray Images

This repository contains code for denoising of special format (not standard shape channel, but interleaved) x-ray images.
Due to the nature of the project, there are no ground truth images available. Therefore we denoise unsupervised. The distribution of the noise has to match the loss function. After carefull study of the physics of the device, we find a L1 loss function to denoise best, as the denoising will approximate the median, not the arithmetic mean.

Contains code for hyperparameter search, architecture search, testing and benchmarking versus traditional denoising methods.

As the denoising was part of a bigger project, files for demosaicing and joint demosaicing and denoising are available as well.
