import numpy as np
import argparse
import tqdm
import numpy as np
import os
import os.path as osp
from glob import glob
from scipy.io import loadmat
import re
import tifffile

def loadFLIMbin(binpath=None, buffer=None, num_bins=3):
    ''' Reads decay matrix from a FLIM .bin file.
        Notes: as of right now does not fully support all decay matrix dimensions.
        Takes a path to a .bin file
        Only works if the 3rd dimension mod binning is equal to 2
        Return a FLIMtuple (decaymatrix, time_res_bin, img_res, time_interval) '''
    dtype = np.int32
    if binpath is not None:
        data = np.fromfile(binpath, dtype)
    elif buffer is not None:
        data = np.frombuffer(buffer, dtype)
    else:
        raise Exception('Has to specify either a binpath or a buffer')
    time_res = data[3]
    img_res = data[0]
    decaymatrix_raw = data[5:].reshape((time_res, img_res, img_res), order='F')
    decaymatrix_raw = decaymatrix_raw.transpose((2,1,0))
    print(time_res)
    # binning
    if num_bins > 1:
        shape = decaymatrix_raw.shape
        shape = (*shape[:-1], num_bins, round((shape[-1]-2)/num_bins))
        n_time_points = shape[-2]*shape[-1]
        decaymatrix = decaymatrix_raw[:,:,time_res-n_time_points:time_res]
        
        decaymatrix = decaymatrix.reshape(shape, order='F')
        decaymatrix = np.mean(decaymatrix, axis=2)
    else:
        decaymatrix = decaymatrix_raw
    
    # more numpythonic this way
    decaymatrix = decaymatrix.transpose([2, 0, 1])
    decaymatrix = np.array(decaymatrix,np.float32)
    
    # decaymatrix /=np.max(decaymatrix)
    #decaymatrix = np.array(decaymatrix*65535,np.int16)
    #sh = list(decaymatrix.shape)
    #tmp = sh[-1]
    #sh[-1] = sh[0]
    #sh[0]=tmp
    #decaymatrix = decaymatrix.reshape(sh)
    time_period_ns = 12.508416*num_bins/time_res;
    time_res_bin = time_res/num_bins;
    print(num_bins)
    print(time_period_ns)
    print(time_res_bin)
    roi = re.findall(r"ROI\d", binpath, re.I)
    roi = roi[0] if len(roi)>0 else ""
    well = re.findall(r"Well\d",binpath, re.I)
    well = well[0] if len(well)>0 else ""
    #print(well)
    roi = roi+well
    integration = re.findall(r"(\d\d|\d\d\d)s", binpath, re.I)
    percent = re.findall(r"(\d|\d.\d)percent", binpath, re.I)
    filename = os.path.basename(os.path.normpath(binpath[:-4]))
    print(filename)
    return {"img":decaymatrix, "time_res_bin":time_res_bin, "res":img_res, "time_period_ns":time_period_ns, "roi":roi, "integration_time":integration,"percent":percent,"filename":filename}

def loadFLIMbins(binspath):
    ''' Reads all FLIM bins in a folder and returns a dictionary where keys
        are file names and values are FLIMTuples '''
    flim_dict = {}
    binfiles = glob(binspath+'/*.bin')
    len_bin_files = len(binfiles)
    
    for filepath in tqdm.tqdm(binfiles):
        flim_dict[osp.basename(filepath)] = loadFLIMbin(filepath)
        
    return flim_dict
    

parser = argparse.ArgumentParser(description='Stack Averager.')

parser.add_argument('--path', metavar='P', type=str, nargs=1,
                    help='Path to bins')
parser.add_argument('--out_path', metavar='P', type=str, nargs=1,
                    help='Path to output')
                    
args = parser.parse_args()

im_dict = loadFLIMbins(args.path[0])
import os
try:
    os.mkdir(os.path.join(args.out_path[0],"X"))
except:
    pass
for i in im_dict:
    folder_name = im_dict[i]["filename"]
    #print(im_dict[i][-4],im_dict[i][-3],im_dict[i][-2],im_dict[i][2:-1])
    print(folder_name)
    try:
        if "100s" in folder_name:
            os.mkdir(os.path.join(args.out_path[0],"y",folder_name))
        else:
            os.mkdir(os.path.join(args.out_path[0],"X",folder_name))
    except OSError as error:
        print(error)
    
    if "100s" in folder_name:
        folder = os.path.join(args.out_path[0],"y",folder_name)
    else:
        folder = os.path.join(args.out_path[0],"X",folder_name)

    roi = im_dict[i]["roi"]
    if roi == "":
        roi = "ref"
    tifffile.imsave(os.path.join(folder,f"{roi}.tif"),im_dict[i]["img"][50:])
















