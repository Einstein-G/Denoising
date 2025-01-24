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

import os
os.environ["CUDA_VISIBLE_DEVICES"] =""
def load_mat_data(path, key = "NADH_n", read_mode = "h5py"):

    from scipy.io import loadmat
    import h5py

    files = list(filter(lambda x: ".mat" in x,os.listdir(path)))

    loaded_keys = []
    skipped_keys = []
    names = []
    X,y = [],[]
    for f in files:
        data = None
        roi = None
        ls = None
        try:
            if(read_mode == "scipy"):
                data = loadmat(os.path.join(path,f))
                ls = list(data.keys())
                roi = data[key]
                
            else:
                data = h5py.File(os.path.join(path,f),'r')
                print(data.keys())
                roi = None
            
                roi = np.array(data[key])   
                
                
        except:
            if(read_mode == "h5py"):
                ls = list(data.keys()) 
                print(ls)      
                data.close()

            k = list(filter(lambda x: "_n" in x,ls))[-1]
            skipped_keys.append(k)

        if roi is not None:
            print(roi.shape)
            if(len(roi.shape) == 4):
               roi = np.moveaxis(roi,0,-1)
            
            loaded_keys.append(key)
            avg = np.mean(roi,axis = -1)
           
            if(len(roi.shape) == 4):
                X.append(roi[:,:,:,0])
            elif len(roi.shape) == 3:
                X.append(roi[:,:,0])
            elif len(roi.shape) == 2:
                X.append(roi)
            
            y.append(avg)
            names.append(f)
    
    X = np.array(X)
    y = np.array(y)

    if len(skipped_keys) == 0:
        skipped_keys.append(None)

    print(f"Loaded {X.shape[0]} 1x frames of shape {X.shape[1:]}")
    print(f"Loaded {len(loaded_keys)} rois of {key} and skipped {len(skipped_keys)} of {skipped_keys[-1]}")

    return X,y, names


parser = argparse.ArgumentParser(description = 'Image denoiser')
parser.add_argument('--config',required= True, type=str, nargs=1,
                    help='Path to config file')


args = parser.parse_args()
print("......................")
print(args.config[0])
f = open(args.config[0],'rb')
config = json.load(f)
print(config)


input_path = config["input_path"]
output_path = config["output_path"]
model_name = config["model_name"]
model_load_path = config["model_load_path"]

X,y, names = load_mat_data(input_path, key = "NADH_n" if "NADH" in model_name else "FAD_n", read_mode = "h5py")

print("----------STARTING----------")
axes = 'YX' if len(X[0].shape) == 2 else "ZYX"
tiles = (1,1) if len(X[0].shape) == 2 else (4,4,4)

print(axes, tiles, X[0].shape)
model = CARE(config=None, name=model_name, basedir=model_load_path)
#'GFPNegativeCells_20191008_model'
denoised = []
for i in tqdm(range(len(X))):
    #x = np.rollaxis(data[i],0,3)
    print(X[i].shape)
    restored = model.predict(X[i], axes, n_tiles= tiles)
    denoised.append(restored)


Path(output_path).mkdir(exist_ok=True)
for i in range(len(names)):
    save_tiff_imagej_compatible(join(output_path,'restored_'+ names[i]), denoised[i], axes)
    if len(y[i].shape)>=2:
        save_tiff_imagej_compatible(join(output_path,'avg_'+ names[i]), y[i], axes)
    save_tiff_imagej_compatible(join(output_path,'noisy_'+ names[i]), X[i], axes)
