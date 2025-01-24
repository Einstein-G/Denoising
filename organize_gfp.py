import tifffile as tif
import numpy as np
import re
import os

#base_path = "/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/GFPNegativeCells/"
base_path = "/media/panagiotis/TOSHIBA EXT/Research/Denoising/Data/GFPNegativeCells_20191114/raw_tifs/"
def is_tif(x):
    return ".tif" in x


fnames = list(filter(is_tif,os.listdir(base_path)))


def is_860_ch00(x):
    return "860" in x and "ch00" in x

def is_860_ch01(x):
    return "860" in x and "ch01" in x

def is_755_ch00(x):
    return "755" in x and "ch00" in x

def is_755_ch01(x):
    return "755" in x and "ch01" in x


fnames_860_ch00 = sorted(list(filter(is_860_ch00,fnames)))
fnames_860_ch01 = sorted(list(filter(is_860_ch01,fnames)))

fnames_755_ch00 = sorted(list(filter(is_755_ch00,fnames)))
fnames_755_ch01 = sorted(list(filter(is_755_ch01,fnames)))

timepoints755 = {}

timepoints860 = {}

timepoints755_ch01 = {}

timepoints860_ch00 = {}

outname1 = ""
outname2 = ""

for i,j in zip(fnames_755_ch00,fnames_755_ch01):
    #print(i,j)
    a = tif.imread(base_path+i)
    b = tif.imread(base_path+j)
    a = np.expand_dims(a,-1)
    b = np.expand_dims(b,-1)
    c = np.concatenate([a,b],axis = -1)
    
    time = re.findall(r"t\d\d",i)[-1]
    
    if time not in timepoints755:
        timepoints755[time] =[(i,j)]
        timepoints755_ch01[time] =[j]
    else:
        timepoints755[time].append((i,j))
        timepoints755_ch01[time].append(j)



for i,j in zip(fnames_860_ch00,fnames_860_ch01):
    #print(i,j)
    a = tif.imread(base_path+i)
    b = tif.imread(base_path+j)
    a = np.expand_dims(a,-1)
    b = np.expand_dims(b,-1)
    c = np.concatenate([a,b],axis = -1)
    
    time = re.findall(r"t\d\d",i)[-1]

    if time not in timepoints860:
        timepoints860[time] = [(i,j)]
        timepoints860_ch00[time] = [i]
    else:
        timepoints860[time].append((i,j))
        timepoints860_ch00[time].append(i)

join = os.path.join

try:
    os.mkdir(join(base_path,"860_ch00"))
    os.mkdir(join(base_path,"755_ch01"))
    os.mkdir(join(base_path,"860"))
    os.mkdir(join(base_path,"755"))

except :
    pass

avg755={}
for t in timepoints755_ch01:
    try:
        os.mkdir(join(base_path,"755_ch01",t))
    except:
        pass
    for img in timepoints755_ch01[t]:
        outname = re.findall(r".*?ROI\d*?_",img)[-1][:-1]
        outname1 = outname[:-5]
        a = tif.imread(join(base_path,img))
        tif.imwrite(join(base_path,"755_ch01",t,outname+".tif"),a)
        roi = re.findall(r"ROI\d*?_",img)[-1][:-1]
        print(roi)
        if roi not in avg755:
            avg755[roi] = a/12
        else:
            avg755[roi]+=a/12

print(avg755.keys())
avg860={}
for t in timepoints860_ch00:
    try:
        os.mkdir(join(base_path,"860_ch00",t))
    except:
        pass
    for img in timepoints860_ch00[t]:
        outname = re.findall(r".*?ROI\d*?_",img)[-1][:-1]
        a = tif.imread(join(base_path,img))
        tif.imwrite(join(base_path,"860_ch00",t,outname+".tif"),a)
        roi = re.findall(r"ROI\d*?_",img)[-1][:-1]
        if roi not in avg860:
            avg860[roi] = a/12
        else:
            avg860[roi]+=a/12



for t in timepoints860:
    try:
        os.mkdir(join(base_path,"860",t))
    except:
        pass
    for img in timepoints860[t]:
        img1,img2 = img
        outname = re.findall(r".*?ROI\d*?_",img1)[-1][:-1]
        a = tif.imread(join(base_path,img1))
        b = tif.imread(join(base_path,img2))
        a = np.expand_dims(a,-1)
        b = np.expand_dims(b,-1)
        c = np.concatenate([a,b],axis = 0)
        tif.imwrite(join(base_path,"860",t,outname+".tif"),c)


for t in timepoints755:
    try:
        os.mkdir(join(base_path,"755",t))
    except:
        pass
    for img in timepoints755[t]:
        img1,img2 = img
        outname = re.findall(r".*?ROI\d*?_",img1)[-1][:-1]
        a = tif.imread(join(base_path,img1))
        b = tif.imread(join(base_path,img2))
        a = np.expand_dims(a,0)
        b = np.expand_dims(b,0)
        c = np.concatenate([a,b],axis = 0)
        tif.imwrite(join(base_path,"755",t,outname+".tif"),np.array(c,dtype = np.float32))


try:
    os.mkdir(join(base_path,"755_avg"))
except :
    pass
for roi in avg755:
    print(roi)
    tif.imwrite(join(base_path,"755_avg","GFPNegativeCells_20191114_"+roi+".tif"),np.array(avg755[roi],dtype = np.float32))

try:
    os.mkdir(join(base_path,"860_avg"))
except :
    pass
for roi in avg860:
    print(roi)
    a = avg860[roi]
    b = np.zeros(shape = a.shape)
    a = np.expand_dims(a,0)
    b = np.expand_dims(b,0)
    c = np.concatenate([b,a],axis = 0)
    tif.imwrite(join(base_path,"860_avg","GFPNegativeCells_20191114_"+roi+".tif"),np.array(c,dtype = np.float32))


