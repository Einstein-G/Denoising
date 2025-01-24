import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

img10s = tif.imread("/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/lifetime_denoising_7_5_2019/flim_images/test/compare/ROI5_10s.tif")
img100s = tif.imread("/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/lifetime_denoising_7_5_2019/flim_images/test/compare/ROI5_100s.tif")
imgr10s = tif.imread("/cluster/home/plympe01/georgakoudi_lab01/plympe01/data/lifetime_denoising_7_5_2019/flim_images/test/compare/restored_ROI5_10s.tif")

m1 = np.mean(img10s,axis = 1)
m2 = np.mean(img100s,axis = 1)
m3 = np.mean(imgr10s,axis = 1)
m1 = np.mean(m1,axis = 1)
m2 = np.mean(m2,axis = 1)
m3 = np.mean(m3,axis = 1)


plt.plot(m1)
plt.savefig("decay10.png")
plt.figure()
plt.plot(m2)
plt.savefig("decay100.png")
plt.figure()
plt.plot(m3)
plt.savefig("decay_r_10.png")
