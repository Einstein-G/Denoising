from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE
import cv2 as cv
from os.path import split
import argparse
from os import listdir
from os.path import isfile, join
from os import walk
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description = 'Image denoiser')
parser.add_argument('--config',required= True, type=str, nargs=1,
                    help='Path to config file')


args = parser.parse_args()
print("......................")
print(args.config[0])
f = open(args.config[0],'rb')
config = json.load(f)
print(config)

if "from_arrays" not in config:
    config["from_arrays"] = False

input_path = config["input_path"]
output_path = config["output_path"]
model_name = config["model_name"]
model_load_path = config["model_load_path"]

f = []
for (dirpath, dirnames, filenames) in walk(input_path):
    f.extend(filenames)
    break
print(listdir(input_path))
data = []

fname = []
for i in f:
    if ".tif" in i or ".tiff" in i:
        fname.append(i)
f = fname

for filename in f:
    x = imread(join(input_path,filename))
    data.append(x)
data = np.array(data)
#print(x.shape)

print("----------STARTING----------")
axes = 'YXC'
model = CARE(config=None, name=model_name, basedir=model_load_path)
#'GFPNegativeCells_20191008_model'
denoised = []
for i in tqdm(range(len(data))):
    x = np.rollaxis(data[i],0,3)
    #print(x.shape)
    restored = model.predict(x, axes, n_tiles= (1,1,1))
    denoised.append(restored)


Path(output_path).mkdir(exist_ok=True)
for i in range(len(f)):
    save_tiff_imagej_compatible(join(output_path,'restored_'+ f[i]), denoised[i], axes)
