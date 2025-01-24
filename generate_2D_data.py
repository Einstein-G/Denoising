from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches, permute_axes, Transform
from csbdeep.utils import axes_check_and_normalize

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import ast

def rotate_gen(axes,times):
    def _generator(inputs):
        result = []
        for x, y, axes_in, mask in inputs:
            axes_in = axes_check_and_normalize(axes_in)
            print(x.shape, y.shape) 
            yield x, y, axes_in, mask
            for i in range(times):
                print(x.shape)
                if len(x.shape) == 3:
                    axs = (1,2)
                else:
                    axs = (0,1)
                x = np.rot90(x, axes = axs)
                y = np.rot90(y,axes = axs)
                print(x.shape, y.shape)
                yield x, y, axes_in, mask   

    return Transform('Rotation Transform %s' % times, _generator, times+1)




def no_norm(patches_x,patches_y, x,y,mask,channel):
    return patches_x, patches_y

def simple_norm(patches_x,patches_y, x,y,mask,channel):
    patches_x/= np.max(x)
    patches_y/= np.max(y)
    return patches_x, patches_y

def area_filter(datas, patch_size, dtype=np.float32):
    
    print(datas.shape)
    def is_full(x):
        tmp = x.copy()/np.max(x)
        #print("here",tmp.shape)
        if np.sum(tmp) > 10:
            
            return True
        
        return False
    keep = np.array([is_full(x) for x in datas])
    return keep


def median_filter(threshold=0.4, percentile=99.9):


    from scipy.ndimage.filters import maximum_filter, median_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        filtered = median_filter(image, patch_size, mode='constant')
        #print(filtered)
        return filtered > threshold * np.percentile(image,percentile)
    return _filter


def make_patches(raw_data,config):
    patch_size = ast.literal_eval(config["patch_size"])
    transforms = ast.literal_eval(config["transforms"])

    true_transforms = {
        "permute_axes_XY":permute_axes("XY"),
        "identity":Transform.identity(),
        "rotate 1": rotate_gen("XY",1),
        "rotate 2": rotate_gen("XY",2),
        "rotate 3": rotate_gen("XY",3)
    }

    transforms = [true_transforms[t] for t in transforms]

    if config["no_norm"]:
        X, Y, XY_axes = create_patches (
            raw_data            = raw_data,
            patch_size          = patch_size,
            n_patches_per_image = config["n_patches_per_image"],
            transforms          = transforms,
            save_file           = config["save_file"],
            normalization= simple_norm,
            patch_filter=median_filter()
        )
    else:
        X, Y, XY_axes = create_patches (
            raw_data            = raw_data,
            patch_size          = patch_size,
            n_patches_per_image = config["n_patches_per_image"],
            transforms          = transforms,
            save_file           = config["save_file"],
            #normalization=
        )

    assert X.shape == Y.shape
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)
    return X, Y, XY_axes


def load_mat_data(path, key = "NADH_n", read_mode = "scipy"):

    from scipy.io import loadmat
    import h5py
    files = list(filter(lambda x: ".mat" in x,os.listdir(path)))

    loaded_keys = []
    skipped_keys = []

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
            if len(roi.shape) == 4:
                roi = np.moveaxis(roi,0,-1)
            print(roi.shape)
            loaded_keys.append(key)
            avg = np.mean(roi,axis = -1)
            for i in range(roi.shape[-1]):
                if(len(roi.shape) == 4):
                    X.append(roi[:,:,:,i])
                else:
                    X.append(roi[:,:,i])
                y.append(avg)
    
    #X = np.array(X)
    #y = np.array(y)

    if len(skipped_keys) == 0:
        skipped_keys.append(None)

    print(f"Loaded {len(X)} 1x frames of shape {X[0].shape}")
    print(f"Loaded {len(loaded_keys)} rois of {key} and skipped {len(skipped_keys)} of {skipped_keys[-1]}")

    return X,y

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Patch generator for 2D images')
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
    if "key" not in config:
        config["key"] = "NADH_n"
        
    if "no_norm" not in config.keys():
            config["no_norm"] = False

    if config["from_arrays"]:
        X,y = load_mat_data(config["basepath"], key =config["key"])
        raw_data = RawData.from_arrays(X,y,config["axes"])

    else:

        source_dirs = os.listdir(os.path.join(config["basepath"],config["source_dir"]))
        source_dirs = [os.path.join(config["source_dir"],s) for s in source_dirs]
        print(source_dirs)

        


        raw_data = RawData.from_folder (
            basepath    = config["basepath"],
            source_dirs = source_dirs,
            target_dir  = config["target_dir"],
            axes        = config["axes"],
        )

    make_patches(raw_data,config)
