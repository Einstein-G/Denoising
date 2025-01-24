import numpy as np
from csbdeep.data import RawData, create_patches, permute_axes, Transform
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import ast
from sklearn.model_selection import KFold


from generate_2D_data import make_patches
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Cross Validator')
    parser.add_argument('--data_config',required= True, type=str, nargs=1,
                        help='Path to config file')


    args = parser.parse_args()
    print("......................")
    print(args.data_config[0])
    f = open(args.data_config[0],'rb')
    data_config = json.load(f)
    print(data_config)

    source_dirs = os.listdir(os.path.join(data_config["basepath"],data_config["source_dir"]))
    source_dirs = [os.path.join(data_config["source_dir"],s) for s in source_dirs]
    print(source_dirs)

    if "no_norm" not in data_config.keys():
        data_config["no_norm"] = False

    raw_data = RawData.from_folder (
        basepath    = data_config["basepath"],
        source_dirs = source_dirs,
        target_dir  = data_config["target_dir"],
        axes        = data_config["axes"],
    )

    print("----------------------------")
    X = []
    y = []
    for i in raw_data[0]():
        X.append(i[0])
        y.append(i[1])
        
    X = np.array(X)
    y = np.array(y)
    
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        raw_fold = RawData.from_arrays(X_train,y_train,data_config["axes"])
        patches = make_patches(raw_fold,data_config)

        print(patches[0].shape)

        print(X_train.shape,y_train.shape)
    


    

