import numpy as np
import torch


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, height=None, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.width = width
        if height == None:
            self.height = width
        else:
            self.height = height
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i, mask_shape=None, shift_right=False):

        phasex = i % self.width
        phasey = (i // self.width) % self.height

        mask = pixel_grid_mask(X[0, 0].shape, self.width, phasex, phasey, self.height)

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

        if shift_right:
            orig_mask = mask.clone()
            mask = torch.zeros_like(mask)
            mask[:,1:] += orig_mask[:, :-1]


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
    def mask_channels(self, x, i, mask_shape_low=None, mask_shape_high=None, demosaicing=False, halfpixel=False):
        x = x.unsqueeze(0)
        net_input = torch.empty_like(x)

        if demosaicing:
            if (i % self.width) % 2 == 0:
                mask_shape_low = 'left'
                mask_shape_high = 'right'
            else:
                mask_shape_low = 'right'
                mask_shape_high = 'left'
            net_input[:, :1, :, :], mask_low = self.mask(x[:, :1, :, :], i, mask_shape=mask_shape_low)
            net_input[:, 1:, :, :], mask_high = self.mask(x[:, 1:, :, :], i, mask_shape=mask_shape_high)
        else:
            if halfpixel:
                rng = np.random.default_rng()
                channel = rng.integers(2)
                if channel == 0:
                    net_input[:, :1, :, :], mask_low = self.mask(x[:, :1, :, :], i, mask_shape=mask_shape_low)
                    net_input[:, 1:, :, :], mask_high = x[:, 1:, :, :], torch.zeros_like(mask_low)
                elif channel == 1:
                    net_input[:, 1:, :, :], mask_high = self.mask(x[:, 1:, :, :], i, mask_shape=mask_shape_low)
                    net_input[:, :1, :, :], mask_low = x[:, :1, :, :], torch.zeros_like(mask_high)
            else:
                net_input[:, :1, :, :], mask_low = self.mask(x[:, :1, :, :], i, mask_shape=mask_shape_low)
                net_input[:, 1:, :, :], mask_high = self.mask(x[:, 1:, :, :], i, mask_shape=mask_shape_high)

        mask = torch.stack([mask_low, mask_high], axis=-3)

        # from eval import plot_tensors
        # plot_tensors([net_input[0], mask])

        return net_input.squeeze(0), mask

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


def pixel_grid_mask(shape, patch_width, phase_x, phase_y, patch_height=None):
    if patch_height == None:
        patch_height = patch_width

    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_height == phase_y and j % patch_width == phase_x):
                A[i, j] = 1
    return torch.Tensor(A)


def interpolate_mask(tensor, mask, mask_inv):

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]

    kernel = torch.Tensor(kernel)
    kernel = kernel / kernel.sum()

    padded_tensor = torch.nn.ReplicationPad2d(1)(tensor)
    padded_inv_mask = torch.ones_like(padded_tensor[0,0])
    padded_inv_mask[1:-1, 1:-1] = mask_inv

    filtered_tensor = torch.nn.functional.conv2d(padded_tensor * padded_inv_mask, kernel, stride=1)

    return filtered_tensor * mask + tensor * mask_inv