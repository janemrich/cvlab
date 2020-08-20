from torch.utils.data import random_split
import numpy as np
import os
import re


def torch_random_split_frac(dataset, fracs):
	assert sum(fracs)==1
	n = len(dataset)
	return random_split(dataset, [round(n*f) for f in fracs])

def to_rgb(hl):
	raise NotImplementedError

def read_pgm(filename, byteorder='>'):
	"""Return image data from a raw PGM file as numpy array.

	Format specification: http://netpbm.sourceforge.net/doc/pgm.html

	"""
	with open(filename, 'rb') as f:
		first = f.readline()
		if first != b'P5\n':
			raise ValueError("pgm mode not supported {} in file {}".format(first, filename))
		(width, height) = [int(i) for i in f.readline().split()]
		depth = int(f.readline())
		assert depth <= 65535

		return np.frombuffer(f.read(),
								dtype='u2' if depth > 255 else 'u1',
								count=int(width)*int(height),
								offset=0
								).reshape((int(height), int(width)))
