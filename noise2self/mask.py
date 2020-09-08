import numpy as np
import torch


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i, mask_shape=None):

        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size

        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)

        if mask_shape:
            orig_mask = mask.clone()
            ### makes mask with two pixels next to each other
            if mask_shape == 'right':
                # mask the pixel on the right as well
                mask[:,1:] += orig_mask[:,:-1]
            elif mask_shape == 'left':
                mask[:,:-1] += orig_mask[:,1:]
            elif mask_shape == 'both':
                mask[:,1:] += orig_mask[:,:-1]
                mask[:,:-1] += orig_mask[:,1:]
            else:
                pass

        mask_inv = torch.ones(mask.shape) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError
            
        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    """
    masks multiple channels

    assumes X has 3 dimensions with channels
    """
    def mask_channels(self, x, i, mask_shape_low=None, mask_shape_high=None):
        x = x.unsqueeze(0)
        net_input = torch.empty_like(x)

        net_input[:, :1, :, :], mask_low = self.mask(x[:, :1, :, :], i, mask_shape=mask_shape_low)
        net_input[:, 1:, :, :], mask_high = self.mask(x[:, 1:, :, :], i, mask_shape=mask_shape_high)
        mask = torch.stack([mask_low, mask_high], axis=-3)

        return x.squeeze(0), net_input.squeeze(0), mask
				

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)


def interpolate_mask(tensor, mask, mask_inv):

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]

    kernel = torch.Tensor(kernel)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(torch.nn.ReplicationPad2d(1)(tensor*mask_inv + torch.full_like(tensor, 0.5) * mask), kernel, stride=1)

    return filtered_tensor * mask + tensor * mask_inv
