from torch.utils.data import random_split

def torch_random_split_frac(dataset, fracs):
	assert sum(fracs)==1
	n = len(dataset)
	return random_split(dataset, [round(n*f) for f in fracs])