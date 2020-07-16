import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
import os
import tifffile
from matplotlib import image
import glob
import random


class SmithData():
	"""Manages access to Smith images dataset format."""

	def __init__(self, root, invert=True, crop=False):
		self.root = root
		self.invert = invert
		self.crop = crop
		self.paths_grouped = self.load_grouped_filenames() # [(high, low, rgb), ...]
		if self.crop:
			self.compute_masks()

	def compute_masks(self):
		t_row = 4.5
		t_col = 7.0
		self.masks = []
		max_height = 0
		max_width = 0
		for path in self.paths_grouped:
			arr = np.array(Image.open(os.path.join(self.root, path[0]))) # use high image to calculate masking box
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
	
		return list(zip(files[0::3], files[1::3], files[2::3]))

	def __getitem__(self, idx):
		high = Image.open(os.path.join(self.root, self.paths_grouped[idx][0]))
		low = Image.open(os.path.join(self.root, self.paths_grouped[idx][1]))
		bbox = self.masks[idx]

		arr = np.stack((np.array(high), np.array(low)), axis=0)

		# normalize to 0-1 range
		arr = arr / 65535.0

		if self.crop:
			arr = arr[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]	

		if self.invert:
			return 1.0 - arr
		else:
			return arr

	def __len__(self):
		return len(self.paths_grouped)


class ProDemosaicDataset(SmithData):

	def __init__(self, root, target_size, invert=True, crop=True, patches_per_image=8):
		super(ProDemosaicDataset, self).__init__(root, invert, crop)
		self.patch_rows = target_size[1]
		self.patch_cols = target_size[0] + 1 # plus one because we extract the high and low patch shifted and need one extra column
		self.patches_per_image = patches_per_image
		self.patches_positions = [[]] * super(ProDemosaicDataset, self).__len__()

	def create_patches(self, idx, pro_shape, patch_shape):
		"""Creates a list of top left points of random patches for image idx and saves them to patches_positions"""
		# cut a random patch from the image	
		shift_row = 0
		shift_col = 0	
		
		diff_row = pro_shape[1] - patch_shape[1]
		diff_col = pro_shape[2] - patch_shape[2]
		
		positions = []
		for _ in range(self.patches_per_image):	
			if diff_row > 0:
				shift_row = random.randrange(diff_row)
			if diff_col > 0:
				shift_col = random.randrange(diff_col)
			positions.append((shift_row, shift_col))
		
		self.patches_positions[idx]= positions


	def __getitem__(self, idx):
		idx_img = idx // self.patches_per_image
		idx_patch = idx % self.patches_per_image
		
		pro = super(ProDemosaicDataset, self).__getitem__(idx_img)
		patch = np.zeros((2, self.patch_rows, self.patch_cols))
		
		if len(self.patches_positions[idx_img]) <= idx_patch:
			# we did not generate the ranndom patch positions for this image yet
			self.create_patches(idx_img, pro.shape, patch.shape) 

		shift_row, shift_col = self.patches_positions[idx_img][idx_patch]
		# if patch is larger than image in a dimension, we make sure to stay in array range
		patch = pro[:,
			shift_row:shift_row+patch.shape[1]-min(pro.shape[1] - patch.shape[1], 0),
			shift_col:shift_col+patch.shape[2]-min(pro.shape[2] - patch.shape[2], 0)]

		patch_high = patch[0, :, :-1]
		patch_low = patch[1, :, 1:]

		sharp = np.zeros((2, self.patch_rows, self.patch_cols-1))

		sharp[0, :, 0::2] = (patch_high[:, 0::2] + patch_high[:, 1::2]) / 2
		sharp[1, :, 1::2] = (patch_low[:, 0::2] + patch_low[:, 1::2]) / 2
		
		sharp = torch.tensor(sharp, dtype=torch.float)
		pro = torch.tensor(patch[:, :, :-1], dtype=torch.float)

		return sharp, pro

	def __len__(self):
		return super(ProDemosaicDataset, self).__len__() * self.patches_per_image

class DemosaicingDataset(Dataset):
	"""Dataset that creates a mosaiced image from an original"""

	def __init__(self, root, target_size=None):
		self.root = root
		self.imgs = list(sorted(os.listdir(root)))
		
		# get sizes
		self.imgs_size = [Image.open(os.path.join(self.root, i), mode='r').size for i in self.imgs]
		self.max_size = (
			max([s[0] for s in self.imgs_size]),
			max([s[1] for s in self.imgs_size])
		)
		self.min_size = (
			min([s[0] for s in self.imgs_size]),
			min([s[1] for s in self.imgs_size])
		)
		self.target_size = target_size
		if target_size is None:
			self.target_size = self.max_size

		self.mosaic_mask = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.float)
		self.mosaic_mask[1::2, 1::2, 0] = 1.0
		self.mosaic_mask[1::2, 0::2, 1] = 1.0
		self.mosaic_mask[0::2, 1::2, 1] = 1.0
		self.mosaic_mask[0::2, 0::2, 2] = 1.0

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		img = Image.open(img_path).convert('RGB')
		img = np.array(ImageOps.fit(img, self.target_size))/255.0
		mosaic = self.mosaic_mask * np.copy(img)
		img = torch.Tensor(img).permute(2, 0, 1)
		mosaic = torch.Tensor(mosaic).permute(2, 0, 1)
		return mosaic, img

	def __len__(self):
		return len(self.imgs)


class DemosaicN2NDataset(Dataset):
	"""Dataset that creates noised mosaic as input image and a demosaiced with different noise as target.
	"""
	def __init__(self, root, target_size=None):
		self.root = root
		self.imgs = list(sorted(os.listdir(root)))
		
		# get sizes
		self.imgs_size = [Image.open(os.path.join(self.root, i), mode='r').size for i in self.imgs]
		self.max_size = (
			max([s[0] for s in self.imgs_size]),
			max([s[1] for s in self.imgs_size])
		)
		self.min_size = (
			min([s[0] for s in self.imgs_size]),
			min([s[1] for s in self.imgs_size])
		)
		self.target_size = target_size
		if target_size is None:
			self.target_size = self.max_size

		self.mosaic_mask = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.float)
		self.mosaic_mask[1::2, 1::2, 0] = 1.0
		self.mosaic_mask[1::2, 0::2, 1] = 1.0
		self.mosaic_mask[0::2, 1::2, 1] = 1.0
		self.mosaic_mask[0::2, 0::2, 2] = 1.0

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		img = Image.open(img_path).convert('RGB')
		img = np.array(ImageOps.fit(img, self.target_size))/255.0
		
		noised1 = np.clip(np.random.normal(loc=img, scale=self.std), 0, 1)
		noised2 = np.clip(np.random.normal(loc=img, scale=self.std), 0, 1)
		
		mosaic = self.mosaic_mask * noised1
		
		x = torch.Tensor(mosaic, dtype=torch.float).permute(2, 0, 1)
		target = torch.Tensor(noised2, dtype=torch.float).permute(2, 0, 1)
		
		return x, target

	def __len__(self):
		return len(self.imgs)


class N2NDataset(Dataset):
	"""Dataset that creates two artificially noised images from an original"""

	def __init__(self, root, target_size=None):
		self.root = root
		self.imgs = list(sorted(os.listdir(root)))
		
		# get sizes
		self.imgs_size = [Image.open(os.path.join(self.root, i), mode='r').size for i in self.imgs]
		self.max_size = (
			max([s[0] for s in self.imgs_size]),
			max([s[1] for s in self.imgs_size])
		)
		self.min_size = (
			min([s[0] for s in self.imgs_size]),
			min([s[1] for s in self.imgs_size])
		)
		self.target_size = target_size
		if target_size is None:
			self.target_size = self.max_size

		self.std = 0.05

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		img = Image.open(img_path).convert('RGB')
		img = np.array(ImageOps.fit(img, self.target_size))/255.0
		
		noised1 = np.clip(np.random.normal(loc=img, scale=self.std), 0, 1)
		noised2 = np.clip(np.random.normal(loc=img, scale=self.std), 0, 1)
		
		x = torch.tensor(noised1, dtype=torch.float).permute(2, 0, 1)
		target = torch.tensor(noised2, dtype=torch.float).permute(2, 0, 1)
		
		return x, target

	def __len__(self):
		return len(self.imgs)


# class N2VDataset(Dataset):
#     """
#     CODE PORTED TO PYTORCH FROM https://github.com/juglab/n2v/internals/N2V_DataWrapper.py
#     """

#     def __init__(self, X, Y, )



class N2VDataGenerator():
	"""
	CODE COPIED FROM https://github.com/juglab/n2v/internals/N2V_DataGenerator.py
	The 'N2V_DataGenerator' enables training and validation data generation for Noise2Void.
	"""

	def load_imgs(self, files, dims='YX'):
		"""
		Helper to read a list of files. The images are not required to have same size,
		but have to be of same dimensionality.

		Parameters
		----------
		files  : list(String)
				 List of paths to tiff-files.
		dims   : String, optional(default='YX')
				 Dimensions of the images to read. Known dimensions are: 'TZYXC'

		Returns
		-------
		images : list(array(float))
				 A list of the read tif-files. The images have dimensionality 'SZYXC' or 'SYXC'
		"""
		assert 'Y' in dims and 'X' in dims, "'dims' has to contain 'X' and 'Y'."

		tmp_dims = dims
		for b in ['X', 'Y', 'Z', 'T', 'C']:
			assert tmp_dims.count(b) <= 1, "'dims' has to contain {} at most once.".format(b)
			tmp_dims = tmp_dims.replace(b, '')

		assert len(tmp_dims) == 0, "Unknown dimensions in 'dims'."

		if 'Z' in dims:
			net_axes = 'ZYXC'
		else:
			net_axes = 'YXC'

		move_axis_from = ()
		move_axis_to = ()
		for d, b in enumerate(dims):
			move_axis_from += tuple([d])
			if b == 'T':
				move_axis_to += tuple([0])
			elif b == 'C':
				move_axis_to += tuple([-1])
			elif b in 'XYZ':
				if 'T' in dims:
					move_axis_to += tuple([net_axes.index(b)+1])
				else:
					move_axis_to += tuple([net_axes.index(b)])
		imgs = []
		for f in files:
			if f.endswith('.tif') or f.endswith('.tiff'):
				imread = tifffile.imread
			elif f.endswith('.png'):
				imread = image.imread
			elif f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.JPG'):
				raise ValueError("JPEG is not supported, because it is not loss-less and breaks the pixel-wise independence assumption.")
			else:
				raise ValueError("Filetype '{}' is not supported.".format(f))

			img = imread(f).astype(np.float32)
			assert len(img.shape) == len(dims), "Number of image dimensions doesn't match 'dims'."

			img = np.moveaxis(img, move_axis_from, move_axis_to)

			if not ('T' in dims):    
				img = img[np.newaxis]

			if not ('C' in dims):
				img = img[..., np.newaxis]

			imgs.append(img)

		return imgs

	def load_imgs_from_directory(self, directory, filter='*.tif', dims='YX'):
		"""
		Helper to read all files which match 'filter' from a directory. The images are not required to have same size,
		but have to be of same dimensionality.

		Parameters
		----------
		directory : String
					Directory from which the data is loaded.
		filter    : String, optional(default='*.tif')
					Filter to match the file names.
		dims      : String, optional(default='YX')
					Dimensions of the images to read. Known dimensions are: 'TZYXC'

		Returns
		-------
		images : list(array(float))
				 A list of the read tif-files. The images have dimensionality 'SZYXC' or 'SYXC'
		"""

		files = glob(os.path.join(directory, filter))
		files.sort()
		return self.load_imgs(files, dims=dims)


	def generate_patches_from_list(self, data, num_patches_per_img=None, shape=(256, 256), augment=True, shuffle=False):
		"""
		Extracts patches from 'list_data', which is a list of images, and returns them in a 'numpy-array'. The images
		can have different dimensionality.

		Parameters
		----------
		data                : list(array(float))
							  List of images with dimensions 'SZYXC' or 'SYXC'
		num_patches_per_img : int, optional(default=None)
							  Number of patches to extract per image. If 'None', as many patches as fit i nto the
							  dimensions are extracted.
		shape               : tuple(int), optional(default=(256, 256))
							  Shape of the extracted patches.
		augment             : bool, optional(default=True)
							  Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
		shuffle             : bool, optional(default=False)
							  Shuffles extracted patches across all given images (data).

		Returns
		-------
		patches : array(float)
				  Numpy-Array with the patches. The dimensions are 'SZYXC' or 'SYXC'
		"""
		patches = []
		for img in data:
			for s in range(img.shape[0]):
				p = self.generate_patches(img[s][np.newaxis], num_patches=num_patches_per_img, shape=shape, augment=augment)
				patches.append(p)

		patches = np.concatenate(patches, axis=0)

		if shuffle:
			np.random.shuffle(patches)

		return patches

	def generate_patches(self, data, num_patches=None, shape=(256, 256), augment=True):
		"""
		Extracts patches from 'data'. The patches can be augmented, which means they get rotated three times
		in XY-Plane and flipped along the X-Axis. Augmentation leads to an eight-fold increase in training data.

		Parameters
		----------
		data        : list(array(float))
					  List of images with dimensions 'SZYXC' or 'SYXC'
		num_patches : int, optional(default=None)
					  Number of patches to extract per image. If 'None', as many patches as fit i nto the
					  dimensions are extracted.
		shape       : tuple(int), optional(default=(256, 256))
					  Shape of the extracted patches.
		augment     : bool, optional(default=True)
					  Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.

		Returns
		-------
		patches : array(float)
				  Numpy-Array containing all patches (randomly shuffled along S-dimension).
				  The dimensions are 'SZYXC' or 'SYXC'
		"""

		patches = self.__extract_patches__(data, num_patches=num_patches, shape=shape, n_dims=len(data.shape)-2)
		if shape[-2] == shape[-1]:
			if augment:
				patches = self.__augment_patches__(patches=patches)
		else:
			if augment:
				print("XY-Plane is not square. Omit augmentation!")

		np.random.shuffle(patches)
		print('Generated patches:', patches.shape)
		return patches

	def __extract_patches__(self, data, num_patches=None, shape=(256, 256), n_dims=2):
		if num_patches == None:
			patches = []
			if n_dims == 2:
				if data.shape[1] > shape[0] and data.shape[2] > shape[1]:
					for y in range(0, data.shape[1] - shape[0] + 1, shape[0]):
						for x in range(0, data.shape[2] - shape[1] + 1, shape[1]):
							patches.append(data[:, y:y + shape[0], x:x + shape[1]])

					return np.concatenate(patches)
				elif data.shape[1] == shape[0] and data.shape[2] == shape[1]:
					return data
				else:
					print("'shape' is too big.")
			elif n_dims == 3:
				if data.shape[1] > shape[0] and data.shape[2] > shape[1] and data.shape[3] > shape[2]:
					for z in range(0, data.shape[1] - shape[0] + 1,  shape[0]):
						for y in range(0, data.shape[2] - shape[1] + 1, shape[1]):
							for x in range(0, data.shape[3] - shape[2] + 1, shape[2]):
								patches.append(data[:, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

					return np.concatenate(patches)
				elif data.shape[1] == shape[0] and data.shape[2] == shape[1] and data.shape[3] == shape[2]:
					return data
				else:
					print("'shape' is too big.")
			else:
				print('Not implemented for more than 4 dimensional (ZYXC) data.')
		else:
			patches = []
			if n_dims == 2:
				for i in range(num_patches):
					y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
																								 data.shape[
																										  2] - shape[
																										  1] + 1)
					patches.append(data[0, y:y + shape[0], x:x + shape[1]])

				if len(patches) > 1:
					return np.stack(patches)
				else:
					return np.array(patches)[np.newaxis]
			elif n_dims == 3:
				for i in range(num_patches):
					z, y, x = np.random.randint(0, data.shape[1] - shape[0] + 1), np.random.randint(0,
																									data.shape[
																											 2] - shape[
																											 1] + 1), np.random.randint(
						0, data.shape[3] - shape[2] + 1)
					patches.append(data[0, z:z + shape[0], y:y + shape[1], x:x + shape[2]])

				if len(patches) > 1:
					return np.stack(patches)
				else:
					return np.array(patches)[np.newaxis]
			else:
				print('Not implemented for more than 4 dimensional (ZYXC) data.')

	def __augment_patches__(self, patches):
		if len(patches.shape[1:-1]) == 2:
			augmented = np.concatenate((patches,
										np.rot90(patches, k=1, axes=(1, 2)),
										np.rot90(patches, k=2, axes=(1, 2)),
										np.rot90(patches, k=3, axes=(1, 2))))
		elif len(patches.shape[1:-1]) == 3:
			augmented = np.concatenate((patches,
										np.rot90(patches, k=1, axes=(2, 3)),
										np.rot90(patches, k=2, axes=(2, 3)),
										np.rot90(patches, k=3, axes=(2, 3))))

		augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
		return augmented
