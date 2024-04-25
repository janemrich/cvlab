import bm3d
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image")

args = parser.parse_args()
im_path = Path(args.image)

im = np.array(Image.open(im_path)) / 65535
print('in')
print(im.max())
print(im.min())
import time

start = time.time()

denoised = bm3d.bm3d(im, np.sqrt(5/65535))

end = time.time()
print('time')
print(end - start)

print('out')
print(denoised.max())
print(denoised.min())

denoised = Image.fromarray((denoised * 65535).astype(np.uint32), mode='I')

denoised.save(im_path.parent / ('bm3d_' + im_path.parts[-1]))