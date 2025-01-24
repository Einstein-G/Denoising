from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import argparse
import json
import os
import pickle as pkl

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config',required= True, type=str, nargs=1,
                    help='Path to config file')


args = parser.parse_args()
print("......................")
print(args.config[0])
f = open(args.config[0],'rb')
config = json.load(f)
print(config)

data_path = config["data_path"]
model_save_path = config["model_save_path"]
model_name = config["model_name"]

if "custom_patches" in config:
    if config["custom_patches"]:
        f = open(data_path,"rb")
        a = pkl.load(f)
        f.close()
        X,Y = a[0]
        X_val,Y_val = a[1]
        axes = a[2]
    else:
        (X,Y), (X_val,Y_val), axes = load_training_data(data_path, validation_split=0.1, verbose=True)
else:
    (X,Y), (X_val,Y_val), axes = load_training_data(data_path, validation_split=0.1, verbose=True)

#print("_________________",axes)
#axes = "SCZYX"
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
plt.savefig("patches.png")

config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=config["train_steps_per_epoch"],train_loss = config['train_loss'],train_epochs = config["train_epochs"], unet_n_depth = config["unet_n_depth"], unet_n_first = config["unet_n_first"], unet_kern_size = config["unet_kern_size"],train_learning_rate = config["train_learning_rate"])
print(config)

model = CARE(config, model_name, basedir=model_save_path)

history = model.train(X,Y, validation_data=(X_val,Y_val))

plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])
plt.savefig("history.png")

plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source')
plt.savefig("validation_patches.png")

