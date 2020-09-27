import sys
import os
import argparse
from PIL import Image
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inpath")
parser.add_argument("outpath")

args = parser.parse_args()
inpath = Path(args.inpath)
outpath = Path(args.outpath)

images = [Image.open(inpath / x) for x in os.listdir(inpath)]
loc = np.array([[int(x[0]), int(x[1])] for x in os.listdir(inpath)])
widths, heights = zip(*(i.size for i in images))

total_width = loc[:,1].max()+1
max_height = loc[:, 0].max()+1

new_im = Image.new('I', (total_width*128, max_height*128))

for im, (row, col) in zip(images, loc):
  new_im.paste(im, (128*col, 128*row))

new_im.save(outpath / ('col_' + inpath.parts[-1] + '.png'))
