import tifffile as tif
import numpy as np
import os
import tqdm




path755 = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/755/X/"
path860 = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/860/X/"
outpath = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/joint/X/"

doX = True
doy =True
if doX:
    fnames755 = sorted(os.listdir(path755))
    fnames860 = sorted(os.listdir(path860))

    for f1,f2 in tqdm.tqdm(zip(fnames755,fnames860)):
        imname1 = sorted(filter(lambda x: ".tif" in x, os.listdir(path755+f1)))
        imname2 = sorted(filter(lambda x: ".tif" in x, os.listdir(path860+f2)))
        for i,j in zip(imname1,imname2):
            
            path1 = os.path.join(path755,f1,i)
            path2 = os.path.join(path860,f2,j)
            ch01 = tif.imread(path1)
            ch02 = tif.imread(path2)
            #ch01= np.expand_dims(ch01,0)
            #ch02= np.expand_dims(ch02,0)
            
            #full_img = np.concatenate([ch01,ch02], axis = 0)
            full_img = np.array([val for pair in zip(ch01, ch02) for val in pair])

            print(full_img.shape)
            outf = os.path.join(outpath,f1+f2)
            outim = os.path.join(outpath,f1+f2,f1[:-3]+"_"+f2[:-3]+".tif")
            if not os.path.exists(outf):
                os.makedirs(outf)
            tif.imsave(outim,full_img)


if doy:

    path755 = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/755/y/"
    path860 = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/860/y/"
    outpath = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/all_hysterectomy/Train/joint/y/"


    fnames755 = sorted(os.listdir(path755))
    fnames860 = sorted(os.listdir(path860))


    for f1,f2 in zip(fnames755,fnames860):

        
        path1 = os.path.join(path755,f1)
        path2 = os.path.join(path860,f2)
        ch01 = tif.imread(path1)
        ch02 = tif.imread(path2)
        #ch01= np.expand_dims(ch01,-1)
        #ch02= np.expand_dims(ch02,-1)

        #full_img = np.concatenate([ch01,ch02], axis = -1)
        full_img = np.array([val for pair in zip(ch01, ch02) for val in pair])
        outim = os.path.join(outpath,f1[:-4]+"_"+f2[:-4]+".tif")
        
        tif.imsave(outim,full_img)


