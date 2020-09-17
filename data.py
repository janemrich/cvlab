import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
import os
import tifffile
from matplotlib import image
import glob
import random
import utils

from noise2self.mask import Masker

class RawDataset():
	"""
	unaltered access to Smith images dataset format 
	"""

	def __init__(self, root, sharp=False, complete_background_noise=False):
		self.root = root
		self.sharp = sharp
		self.paths_grouped = self.load_grouped_filenames() # [(high, low, rgb, ...)]
		self.complete_background_noise = complete_background_noise

	def load_grouped_filenames(self):
		files = sorted(os.listdir(self.root))
		if self.sharp:
			return list(zip(files[0::5], files[1::5], files[4::5]))
		else:
			return list(zip(files[0::3], files[1::3], files[2::3]))

	def __getitem__(self, idx):
		high = Image.open(os.path.join(self.root, self.paths_grouped[idx][0]))
		low = Image.open(os.path.join(self.root, self.paths_grouped[idx][1]))

		arr = np.stack((np.array(high), np.array(low)), axis=0)

		if self.complete_background_noise:
			maxval = 65535
			offset = 10
			black_mask = np.nonzero(arr >= (maxval-offset))
			random_background = np.abs(np.random.normal(0, scale=700, size=arr.shape)) + maxval - offset
			arr[black_mask] = random_background[black_mask]

			minval = 0
			offset = 10
			white_mask = np.nonzero((arr) <= (minval+offset))
			random_background = (- np.abs(np.random.normal(0, scale=30, size=arr.shape))) + offset + 1
			arr[white_mask] = random_background[white_mask]

		return arr

	def get_rgb(self, idx):
		arr = np.array(Image.open(os.path.join(self.root, self.paths_grouped[idx][2])))
		arr /= 255.0
		return arr

	def __len__(self):
		return len(self.paths_grouped)

class SmithData():
	"""Manages access to Smith images dataset format.
	sharp: data source are images from sharp machine
	"""

	def __init__(self, root, invert=True, crop=False, sharp=False, has_rgb=True, complete_background_noise=False):
		self.root = root
		self.has_rgb = has_rgb
		self.invert = invert
		self.crop = crop
		self.sharp = sharp
		self.paths_grouped = self.load_grouped_filenames() # [(high, low, rgb), ...]
		self.remove_plain_pgm()
		if self.crop:
			self.compute_masks()
		self.complete_background_noise = complete_background_noise

	def remove_plain_pgm(self):
		invalid = []
		for i, p in enumerate(self.paths_grouped):
			if p[0].split('.')[1] == 'pgm':
				with open(os.path.join(self.root, p[0]), 'rb') as f:
					if f.readline()  != b'P5\n':
						invalid.insert(0, i)
		for i in invalid:
			del self.paths_grouped[i]

	def compute_masks(self):
		t_row = 5.5
		t_col = 7.5
		self.masks = []
		max_height = 0
		max_width = 0
		for path in self.paths_grouped:
			im_path = os.path.join(self.root, path[0])
			if im_path.split('.')[1] == 'pgm':
				arr = utils.read_pgm(im_path)
			else:
				arr = np.array(Image.open(im_path)) # use high image to calculate masking box
			
			arr_i = 1.0 - (arr / 65535)
			hist_row = np.where(np.sum(arr_i, axis=1) > t_row)
			hist_col = np.where(np.sum(arr_i, axis=0) > t_col)
			bbox = [np.min(hist_row), np.max(hist_row)+1, np.min(hist_col), np.max(hist_col)+1]
			self.masks.append(bbox)
			max_height = max(max_height, bbox[1]-bbox[0])
			max_width = max(max_width, bbox[3]-bbox[2])

		print("Precomupted cropping masks. max_height: {}, max_width: {}".format(max_height, max_width))

	def load_grouped_filenames(self):
		files = sorted(os.listdir(self.root))
		if self.sharp:
			# 0=high, 1=low
			return list(zip(files[0::5], files[1::5], files[4::5]))
		elif self.has_rgb:
			return list(zip(files[0::3], files[1::3], files[2::3]))
		else: # only high and low
			return list(zip(files[0::2], files[1::2]))

	def __getitem__(self, idx):
		high_path = os.path.join(self.root, self.paths_grouped[idx][0])
		low_path = os.path.join(self.root, self.paths_grouped[idx][1])
		
		if high_path.split('.')[1] == 'pgm':
			high = utils.read_pgm(high_path)
			low = utils.read_pgm(low_path)
		else:
			high = np.array(Image.open(high_path)) # use high image to calculate masking box
			low = np.array(Image.open(low_path)) # use high image to calculate masking box
		
		bbox = self.masks[idx] if self.crop else [0, min(low.shape[0], high.shape[0]), 0, min(low.shape[1], high.shape[1])]

		arr = np.stack((high, low), axis=0)

		if self.complete_background_noise:
			maxval = 65535
			offset = 10
			black_mask = np.nonzero(arr >= (maxval-offset))
			random_background = np.abs(np.random.normal(0, scale=700, size=arr.shape)) + maxval - offset
			arr[black_mask] = random_background[black_mask]

			minval = 0
			offset = 10
			white_mask = np.nonzero((arr) <= (minval+offset))
			random_background = (- np.abs(np.random.normal(0, scale=30, size=arr.shape))) + offset + 1
			arr[white_mask] = random_background[white_mask]

		if np.max(arr) < 256:
			arr = arr / 256
		else:
			# normalize to 0-1 range
			arr = arr / 65535.0

		if self.crop:
			arr = arr[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]	

		if self.invert:
			return 1.0 - arr
		else:
			return arr

	def get_rgb(self, idx):
		arr = np.array(Image.open(os.path.join(self.root, self.paths_grouped[idx][2]))).astype('float')
		arr /= 255.0
		bbox = self.masks[idx]
		
		if self.crop:
			arr = arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]	
		
		from matplotlib.colors import rgb_to_hsv
		arr = rgb_to_hsv(arr)
		
		arr = np.transpose(arr, (2, 0, 1))
		return arr

	def __len__(self):
		return len(self.paths_grouped)



class N2SDataset(SmithData):

	def __init__(self, root, target_size, sharp=False, invert=True, crop=True, drop_background=True, patches_per_image=8,
				complete_background_noise=False, channels=2, mask_grid_size=4, mask_shape_low=None, mask_shape_high=None):
		super(N2SDataset, self).__init__(root, invert, crop, sharp, complete_background_noise=complete_background_noise)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 1 # plus one because we extract the high and low patch shifted and need one extra column
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(N2SDataset, self).__len__()
		self.drop_background = drop_background
		self.channels = channels
		self.mask_grid_size = mask_grid_size
		self.mask_shape_high = mask_shape_high
		self.mask_shape_low = mask_shape_low
		self.get_calls = 0


	def create_patches(self, idx, image, images_shape, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		
		diff_row = images_shape[1] - patch_shape[1]
		diff_col = images_shape[2] - patch_shape[2]
		
		positions = []
		fail_count = 0
		max_fails = 10

		# get same patches in test mode
		if self.test:
			random.seed(42)
		while len(positions) < self.patches_per_image:	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			if self.drop_background and np.mean(image[0, shift_row:shift_row+patch_shape[0], shift_col:shift_col+patch_shape[1]]) < 0.1 and fail_count < max_fails:
				fail_count += 1
				continue
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions

	def reset(self):
		self.patches_positions = [[]] * super(N2SDataset, self).__len__()

	def __getitem__(self, idx):
		self.get_calls += 1
		if self.get_calls > self.__len__():
			self.reset()

		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		images = super(N2SDataset, self).__getitem__(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		
		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the random patch positions for this image yet
			self.create_patches(idx_img, images, images.shape, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = images[:,
			shift_row:shift_row+patch.shape[1]-min(images.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(images.shape[2] - patch.shape[2], 0)]

		# images = patch[:, :, :-1]
		images = torch.tensor(patch[:, :, :-1], dtype=torch.float)

		rng = np.random.default_rng()
		masked_pixel = rng.integers(self.mask_grid_size**2)

		masker = Masker(width = self.mask_grid_size, mode='interpolate')
		if self.channels == 1:
			return images, masker.mask(images, masked_pixel, mask_shape_low=self.mask_shape_low, mask_shape_high=self.mask_shape_high)
		if self.channels == 2:
			net_input, mask = masker.mask_channels(images, masked_pixel, mask_shape_low=self.mask_shape_low, mask_shape_high=self.mask_shape_high)
			# from eval import plot_tensors
			# plot_tensors([images, net_input, mask])
			return images, net_input, mask

		return images[:self.channels,:,:], 

	def __len__(self):
		return super(N2SDataset, self).__len__() * self.patches_per_image


class N2SProDemosaicDataset(SmithData):
	"""
		Args:
			fill_missing: 'zero', 'same' or 'interp'
	"""
	def __init__(self, root, target_size, invert=True, crop=True, patches_per_image=8, drop_background=True, renewing_patches=True, fill_missing='same', has_rgb=True, sharp=False,
					complete_background_noise=False, mask_grid_size=4, mask_shape_sharp_low=None, mask_shape_sharp_high=None, mask_shape_pro_low=None, mask_shape_pro_high=None, loss_shape='full', subpixelmask=False, halfpixel=False):
		super(N2SProDemosaicDataset, self).__init__(root, invert, crop, sharp=sharp, has_rgb=has_rgb, complete_background_noise=complete_background_noise)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 3 # plus one because we extract the high and low patch shifted and need one extra column #### and plus two to generate sharp
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(N2SProDemosaicDataset, self).__len__()
		self.fill_missing=fill_missing

		self.drop_background = drop_background
		self.renewing_patches = renewing_patches
		self.get_calls = 0
		# denoising
		self.mask_grid_size = mask_grid_size
		self.mask_shape_sharp_high = mask_shape_sharp_high
		self.mask_shape_sharp_low = mask_shape_sharp_low
		self.mask_shape_pro_high = mask_shape_pro_high
		self.mask_shape_pro_low = mask_shape_pro_low
		self.loss_shape = loss_shape
		self.subpixelmask = subpixelmask
		self.halfpixel = halfpixel

		self.deterministic = False


	def create_patches(self, idx, pro, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		pro_shape = pro.shape

		diff_row = pro_shape[1] - patch_shape[1]
		diff_col = pro_shape[2] - patch_shape[2]

		if self.deterministic:
			random.seed(42)
		
		positions = []
		fail_count = 0
		max_fails = 10
		while len(positions) < self.patches_per_image:	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			if self.drop_background and np.mean(pro[0, shift_row:shift_row+patch_shape[1], shift_col:shift_col+patch_shape[2]]) < 0.05 and fail_count < max_fails:
				fail_count += 1
				continue
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions
		# self.patches_positions = [[(400, 200)*self.patches_per_image], [(400, 200)*self.patches_per_image]]

	def reset(self):
		self.patches_positions = [[]] * super(N2SProDemosaicDataset, self).__len__()
	
	def gen_sharp(self, patch):
		if patch.shape[-1] % 2 == 0:
			patch = patch[:, :, :-1]

		if self.subpixelmask:
			rng = np.random.default_rng()
			masked_pixel = rng.integers(self.mask_grid_size**2)
			masker = Masker(width=self.mask_grid_size, mode='interpolate')
			net_input, mask = masker.mask_channels(patch, masked_pixel, halfpixel=self.halfpixel)
			sharp = gen_normal_sharp(net_input, self.fill_missing)
			return sharp[:,:,1:], mask[:,:,2:-1]

		patch_high = patch[1, :, :-1]
		patch_low = patch[0, :, 1:]

		sharp_sparse = torch.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)))
		sharp = torch.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)*2))

		sharp_sparse[1, :, :] = (patch_high[:, 0::2] + patch_high[:, 1::2]) / 2
		sharp_sparse[0, :, :] = (patch_low[:, 0::2] + patch_low[:, 1::2]) / 2
		
		#random masking
		#to right is wether right high pixel should be masked or the one to the left
		rng = np.random.default_rng()
		masked_pixel = rng.integers(self.mask_grid_size**2)
		to_right = rng.integers(2)

		# masking
		sharp_sparse = sharp_sparse.unsqueeze(0)
		masker = Masker(width=self.mask_grid_size, mode='interpolate')
		sharp_sparse[:,:1], mask_low_sparse = masker.mask(sharp_sparse[:,:1], masked_pixel)
		sharp_sparse[:,1:], mask_high_sparse = masker.mask(sharp_sparse[:,1:], masked_pixel, shift_right=to_right)
		mask_sparse = torch.stack((mask_low_sparse, mask_high_sparse), axis=-3)
		sharp_sparse = sharp_sparse.squeeze(0)

		# now that pixel are masked, go to higher resolution
		sharp[1, :, 0::2] = sharp_sparse[1, :, :]
		sharp[0, :, 1::2] = sharp_sparse[0, :, :]

		# from eval import plot_tensors
		# plot_tensors([sharp])

		# copy pixels
		if self.fill_missing == 'same':
			filled_sharp = torch.zeros_like(sharp)
			filled_sharp += sharp
			filled_sharp[:, :, 1:] += sharp[:, :, :-1]
			sharp = filled_sharp

		elif self.fill_missing == 'interp':
			raise NotImplementedError
		
		else:
			raise ValueError("Unknown fill value {}".format(self.fill_missing))

		# also make mask in higher resolution
		full_mask = torch.zeros_like(sharp)
		full_mask[1, :, 0::2] += mask_sparse[1, :, :]
		full_mask[0, :, 1::2] += mask_sparse[0, :, :]
		fullest_mask = torch.zeros_like(full_mask)
		fullest_mask += full_mask
		fullest_mask[:, :, 1:] += full_mask[:, :, :-1]
		full_mask = fullest_mask

		# only use center masked pixel or all three pixels for the loss
		if self.loss_shape == 'center':
			center = (full_mask[0] + full_mask[1]) == torch.full_like(full_mask[0], 2.0)
			mask = torch.zeros_like(sharp)
			mask[:, center] = 1
		elif self.loss_shape == 'full':
			mask = full_mask

		# from eval import plot_sharp_masking
		# plot_sharp_masking(patch, patch_low, patch_high, sharp_sparse, sharp)
		
		# from eval import plot_tensors
		# plot_tensors([mask_sparse, full_mask, mask])
		
		del sharp_sparse, mask_sparse, filled_sharp, full_mask, fullest_mask
		# plot_tensors([sharp, mask])
		return sharp[:, :, 2:], mask


	def __getitem__(self, idx):
		if self.renewing_patches and not self.deterministic:
			self.get_calls += 1
			if self.get_calls > self.__len__():
				self.reset()

		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		pro = super(N2SProDemosaicDataset, self).__getitem__(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		
		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the random patch positions for this image yet
			self.create_patches(idx_img, pro, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = pro[:,
			shift_row:shift_row+patch.shape[1]-min(pro.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(pro.shape[2] - patch.shape[2], 0)]

		patch = torch.tensor(patch, dtype=torch.float)

		# sharp = torch.tensor(self.gen_sharp(patch), dtype=torch.float)
		net_input, mask = self.gen_sharp(patch)
		net_input = torch.tensor(net_input, dtype=torch.float)
		sharp = gen_normal_sharp(patch, self.fill_missing)[:, :, 1:]
		sharp = torch.tensor(sharp, dtype=torch.float)
		pro = patch[:, :, 2:-1]

		from eval import plot_tensors
		plot_tensors([pro, net_input, mask, sharp, np.abs(net_input-sharp)*10], v=True)

		return pro, net_input, mask, sharp

	def get_full(self, idx):
		"""Does not do patching."""
		pro = super(N2SProDemosaicDataset, self).__getitem__(idx)
		sharp = self.gen_sharp(pro)
		pro = torch.tensor(pro[:, :, :sharp.shape[-1]], dtype=torch.float)
		sharp = torch.tensor(sharp, dtype=torch.float)

		return sharp, pro, super(N2SProDemosaicDataset, self).get_rgb(idx)

	def __len__(self):
		return super(N2SProDemosaicDataset, self).__len__() * self.patches_per_image


class ProDemosaicDataset(SmithData):
	"""
		Args:
			fill_missing: 'zero', 'same' or 'interp'
	"""
	def __init__(self, root, target_size, invert=True, crop=True, patches_per_image=8, fill_missing='same', has_rgb=True):
		super(ProDemosaicDataset, self).__init__(root, invert, crop, has_rgb=has_rgb)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 1 # plus one because we extract the high and low patch shifted and need one extra column
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(ProDemosaicDataset, self).__len__()
		self.fill_missing=fill_missing

	def create_patches(self, idx, pro, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		pro_shape = pro.shape

		diff_row = pro_shape[1] - patch_shape[1]
		diff_col = pro_shape[2] - patch_shape[2]
		
		positions = []
		fail_count = 0
		max_fails = 10
		while len(positions) < self.patches_per_image:	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			if np.mean(pro[0, shift_row:shift_row+patch_shape[1], shift_col:shift_col+patch_shape[2]]) < 0.05 and fail_count < max_fails:
				fail_count += 1
				continue
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions
		# self.patches_positions = [[(400, 200)*self.patches_per_image], [(400, 200)*self.patches_per_image]]

	def reset(self):
		self.patches_positions = [[]] * super(ProDemosaicDataset, self).__len__()
	
	def gen_sharp(self, patch):
		if patch.shape[-1] % 2 == 0:
			patch = patch[:, :, :-1]
		patch_high = patch[1, :, :-1]
		patch_low = patch[0, :, 1:]

		sharp_sparse = np.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)))
		sharp = np.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)*2))

		sharp_sparse[1, :, :] = (patch_high[:, 0::2] + patch_high[:, 1::2]) / 2
		sharp_sparse[0, :, :] = (patch_low[:, 0::2] + patch_low[:, 1::2]) / 2
		
		sharp[1, :, 0::2] = sharp_sparse[1, :, :]
		sharp[0, :, 1::2] = sharp_sparse[0, :, :]

		if self.fill_missing == 'same':
			sharp[:, :, 1:] += sharp[:, :, :-1]

		elif self.fill_missing == 'interp':
			raise NotImplementedError
		
		else:
			raise ValueError("Unknown fill value {}".format(self.fill_missing))
		
		return sharp[:, :, 1:]


	def __getitem__(self, idx):
		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		pro = super(ProDemosaicDataset, self).__getitem__(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		
		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the random patch positions for this image yet
			self.create_patches(idx_img, pro, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = pro[:,
			shift_row:shift_row+patch.shape[1]-min(pro.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(pro.shape[2] - patch.shape[2], 0)]

		patch = torch.tensor(patch, dtype=torch.float)

		sharp = torch.tensor(self.gen_sharp(patch), dtype=torch.float)
		pro = torch.tensor(patch, dtype=torch.float)

		pro = pro[:, :, 2:-1]
		sharp = sharp[:, :, 1:]

		return sharp, pro

	def get_full(self, idx):
		"""Does not do patching."""
		pro = super(ProDemosaicDataset, self).__getitem__(idx)
		sharp = self.gen_sharp(pro)
		pro = torch.tensor(pro[:, :, :sharp.shape[-1]], dtype=torch.float)
		sharp = torch.tensor(sharp, dtype=torch.float)

		return sharp, pro, super(ProDemosaicDataset, self).get_rgb(idx)

	def __len__(self):
		return super(ProDemosaicDataset, self).__len__() * self.patches_per_image


class ProColoringDataset(SmithData):

	def __init__(self, root, target_size, invert=True, crop=True, patches_per_image=8, fill_missing='same'):
		super(ProColoringDataset, self).__init__(root, invert, crop, has_rgb=True)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 1 # plus one because we extract the high and low patch shifted and need one extra column
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(ProColoringDataset, self).__len__()
		self.fill_missing=fill_missing

	def create_patches(self, idx, pro, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		pro_shape = pro.shape

		diff_row = pro_shape[1] - patch_shape[1]
		diff_col = pro_shape[2] - patch_shape[2]
		
		positions = []
		fail_count = 0
		max_fails = 10
		while len(positions) < self.patches_per_image:	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			if np.mean(pro[0, shift_row:shift_row+patch_shape[1], shift_col:shift_col+patch_shape[2]]) < 0.05 and fail_count < max_fails:
				fail_count += 1
				continue
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions
		# self.patches_positions = [[(400, 200)*self.patches_per_image], [(400, 200)*self.patches_per_image]]

	def reset(self):
		self.patches_positions = [[]] * super(ProColoringDataset, self).__len__()

	def __getitem__(self, idx):
		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		pro = super(ProColoringDataset, self).__getitem__(idx_img)
		rgb = super(ProColoringDataset, self).get_rgb(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		assert pro.shape[1] == rgb.shape[1] and pro.shape[2] == rgb.shape[2], (pro.shape, rgb.shape)	

		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the random patch positions for this image yet
			self.create_patches(idx_img, pro, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = pro[:,
			shift_row:shift_row+patch.shape[1]-min(pro.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(pro.shape[2] - patch.shape[2], 0)]
		
		patch_rgb = rgb[:,
			shift_row:shift_row+patch.shape[1]-min(rgb.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(rgb.shape[2] - patch.shape[2], 0)]

		pro = torch.tensor(patch, dtype=torch.float)
		rgb = torch.tensor(patch_rgb, dtype=torch.float)

		return pro, rgb

	def get_full(self, idx):
		"""Does not do patching."""
		pro = super(ProDemosaicRGBDataset, self).__getitem__(idx)
		sharp = self.gen_sharp(pro)
		pro = torch.tensor(pro[:, :, :sharp.shape[-1]], dtype=torch.float)
		sharp = torch.tensor(sharp, dtype=torch.float)

		return sharp, pro, super(ProColoringDataset, self).get_rgb(idx)

	def __len__(self):
		return super(ProColoringDataset, self).__len__() * self.patches_per_image


class ProDemosaicRGBDataset(SmithData):
	"""
		Args:
			fill_missing: 'zero', 'same' or 'interp'
	"""
	def __init__(self, root, target_size, invert=True, crop=True, patches_per_image=8, fill_missing='same', has_rgb=True):
		super(ProDemosaicRGBDataset, self).__init__(root, invert, crop, has_rgb=has_rgb)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 1 # plus one because we extract the high and low patch shifted and need one extra column
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(ProDemosaicRGBDataset, self).__len__()
		self.fill_missing=fill_missing

	def create_patches(self, idx, pro, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		pro_shape = pro.shape

		diff_row = pro_shape[1] - patch_shape[1]
		diff_col = pro_shape[2] - patch_shape[2]
		
		positions = []
		fail_count = 0
		max_fails = 10
		while len(positions) < self.patches_per_image:	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			if np.mean(pro[0, shift_row:shift_row+patch_shape[1], shift_col:shift_col+patch_shape[2]]) < 0.05 and fail_count < max_fails:
				fail_count += 1
				continue
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions
		# self.patches_positions = [[(400, 200)*self.patches_per_image], [(400, 200)*self.patches_per_image]]

	def reset(self):
		self.patches_positions = [[]] * super(ProDemosaicRGBDataset, self).__len__()
	
	def gen_sharp(self, patch):
		if patch.shape[-1] % 2 == 0:
			patch = patch[:, :, :-1]
		patch_high = patch[0, :, :-1]
		patch_low = patch[1, :, 1:]

		sharp = np.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)*2))

		sharp[0, :, 0::2] = (patch_high[:, 0::2] + patch_high[:, 1::2]) / 2
		sharp[1, :, 1::2] = (patch_low[:, 0::2] + patch_low[:, 1::2]) / 2
		
		if self.fill_missing == 'same':
			sharp[0, :, 1::2] = sharp[0, :, 0::2]
			sharp[1, :, 0::2] = sharp[1, :, 1::2]

		elif self.fill_missing == 'interp':
			raise NotImplementedError
		
		else:
			raise ValueError("Unknown fill value {}".format(self.fill_missing))
		
		return sharp

	def __getitem__(self, idx):
		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		pro = super(ProDemosaicRGBDataset, self).__getitem__(idx_img)
		rgb = super(ProDemosaicRGBDataset, self).get_rgb(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		
		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the random patch positions for this image yet
			self.create_patches(idx_img, pro, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = pro[:,
			shift_row:shift_row+patch.shape[1]-min(pro.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(pro.shape[2] - patch.shape[2], 0)]
		
		patch_rgb = rgb[:,
			shift_row:shift_row+patch.shape[1]-min(rgb.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(rgb.shape[2] - patch.shape[2], 0)]

		sharp = torch.tensor(self.gen_sharp(patch), dtype=torch.float)
		rgb = torch.tensor(patch_rgb[:, :, :-1], dtype=torch.float)

		return sharp, rgb

	def get_full(self, idx):
		"""Does not do patching."""
		pro = super(ProDemosaicRGBDataset, self).__getitem__(idx)
		sharp = self.gen_sharp(pro)
		pro = torch.tensor(pro[:, :, :sharp.shape[-1]], dtype=torch.float)
		sharp = torch.tensor(sharp, dtype=torch.float)

		return sharp, pro, super(ProDemosaicRGBDataset, self).get_rgb(idx)

	def __len__(self):
		return super(ProDemosaicRGBDataset, self).__len__() * self.patches_per_image


def gen_normal_sharp(patch, fill_missing):
	if patch.shape[-1] % 2 == 0:
		patch = patch[:, :, :-1]
	# patch_high = patch[1, :, :-1]
	patch_high = patch[1, :, :-1]
	patch_low = patch[0, :, 1:]

	sharp_sparse = np.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)))
	sharp = np.zeros((patch.shape[0], patch.shape[1], (patch.shape[2]//2)*2))

	sharp_sparse[1, :, :] = (patch_high[:, 0::2] + patch_high[:, 1::2]) / 2
	sharp_sparse[0, :, :] = (patch_low[:, 0::2] + patch_low[:, 1::2]) / 2
	
	sharp[1, :, 0::2] = sharp_sparse[1, :, :]
	sharp[0, :, 1::2] = sharp_sparse[0, :, :]

	if fill_missing == 'same':
		sharp[:, :, 1:] += sharp[:, :, :-1]

	elif fill_missing == 'interp':
		raise NotImplementedError
	
	else:
		raise ValueError("Unknown fill value {}".format(self.fill_missing))
	
	return sharp[:, :, 1:]
