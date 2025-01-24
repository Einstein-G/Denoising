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


np.random.seed(10)

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
axes = None
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

def prep_indeces(original):
    ls = []
    indeces = []
    for key,val in original.items():
        if type(val) is list:
            ls.append(val)
    print(ls)
    return ls

def build_config(original,values, epochs = 10):
    cfg = {}
    count = 0
    for key,val in original.items():
        if type(val) is list:
            cfg[key] = values[count]
            count+=1
        else:
            cfg[key] = val
    cfg["train_epochs"] = epochs
    return cfg
            


def build_model(cfg):

    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=cfg["train_steps_per_epoch"],train_loss = cfg['train_loss'],train_epochs = cfg["train_epochs"], unet_n_depth = cfg["unet_n_depth"], unet_n_first = cfg["unet_n_first"], unet_kern_size = cfg["unet_kern_size"],train_learning_rate = cfg["train_learning_rate"])
    print(config)

    model = CARE(config, model_name, basedir=model_save_path)
    return model

options = prep_indeces(config)

from itertools import product
def next_config(original,options):
    p = product(*options)
    
    for i in p:
        config=  build_config(original,i, epochs = 5)
        yield config

def random_config(original,options, n= 15):
    p = list(product(*options))
    
    for i in range(n):
        c = p[np.random.randint(len(p))]
        config=  build_config(original,c, epochs = 10)
        yield config
        
config_score = []

for i,c in enumerate(random_config(config,options, n = 5)):
    
    model = build_model(c)
    history = model.train(X,Y, validation_data=(X_val,Y_val))
    config_score.append((c,history.history["val_loss"][-1]))
    del model


config_score = sorted(config_score, key = lambda x: x[1])

best_config = config_score[0][0]
best_config["train_epochs"] = config["train_epochs"]

f = open(f"best_config_{best_config['model_name']}",'w')
json.dump(best_config,f)
f.close()
exit()

model = build_model(best_config)



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

