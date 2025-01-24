import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import ast
import tifffile
import numpy as np
from beta_analysis import process_image
import tqdm

def mse(y,x):

    if len(y.shape)==3:
        values = []
        for i,j in zip(x,y):
            values.append(np.mean((i-j)**2))
        values=  np.array(values)
        return values
    else:
        return np.mean((x-y)**2)
def mae(y,x):
    if len(y.shape)==3:
        values = []
        for i,j in zip(x,y):
            values.append(np.mean(np.abs(i-j)))
        values=  np.array(values)
        return values
    else:
        return np.mean(np.abs(x-y))


def psnr(y,x):
    mx = 1#np.max([np.max(x),np.max(y)])
    return 20*np.log(mx) - 10* np.log(mse(x,y))

def beta(y,x):
    print(x.shape)
    betas = process_image(x)

    return betas

def compare(files,metrics,labels):
    
    #metric ={m:{l:[] for l in labels} for m in metrics}
    metric ={m:{l:None for l in labels} for m in metrics}
    '''
    for m in metrics:
        for i in range(files["clean"].shape[0]):
            for l in labels:
                if l=='clean' and m!="beta":
                    continue
            
                mtr = eval(m)
                metric[m][l].append(mtr(files["clean"][i],files[l][i]))
    '''
    for m in tqdm.tqdm(metrics):
        for l in labels:
            if l=='clean' and m!="beta":
                continue
        
            mtr = eval(m)
            metric[m][l] = mtr(files["clean"],files[l])

    return metric

def load_and_compare(config):
    
    print(config)

    base_path = config["base_path"]
    filenames = ast.literal_eval(config["filenames"])
    labels = ast.literal_eval(config["file_labels"])

    metrics = ast.literal_eval(config["metrics"])
    plotname = "metrics.png"
    if "plotname" in config:
        plotname=  config["plotname"]

    files = {}
    for i,f in enumerate(filenames):
        img = tifffile.imread(os.path.join(base_path,f))
        files[labels[i]] = img/np.max(img)

    metric  = compare(files,metrics,files.keys())
    print(metric)
    fig,ax = plt.subplots(ncols=len(metrics),nrows=1,figsize = (12,7))
    for i,m in tqdm.tqdm(enumerate(metrics)):
        legend = []
        for l in labels:
                if l=='clean' and m!="beta":
                    continue
                ax[i].scatter(0,metric[m][l])
                legend.append(l)
                
        ax[i].legend(legend)
        ax[i].set_title(m.upper())

    fig.savefig(os.path.join(base_path,plotname))
    return metric
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare stacks')
    parser.add_argument('--config',required= True, type=str, nargs=1,
                    help='Path to config file')
    args = parser.parse_args()
    print("......................")
    print(args.config[0])
    f = open(args.config[0],'rb')
    config = json.load(f)
    metrics = load_and_compare(config)

    if "save_metrics" not in config:
        config["save_metrics"] = "./metrics.csv"
        
    
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv(config["save_metrics"])

    plt.show()