# -*- coding:utf-8 -*-
import numpy as np
from glob import glob
import cv2
import numpy
import math

def psnr_(im1, im2):
    diff = np.abs(im1-im2)
    rmse = np.sum((diff)**2)/(im1.shape[0]*im2.shape[1])
    psnr = 20*np.log10(255/rmse)
    return psnr

def psnr(target, ref):
    import cv2
    target_data = numpy.array(target, dtype=numpy.float64)
    ref_data = numpy.array(ref,dtype=numpy.float64)

    diff = ref_data - target_data
#    print(diff.shape)
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255 / rmse)


gdl_gen=sorted(glob('../Save/Images/gt-lt14-300000/Tests/Step_0/0/gen_*.png'))
gdl_gt=sorted(glob('../Save/Images/gt-lt14-300000/Tests/Step_0/0/gt_*.png'))
l1_gen=sorted(glob('../ccom/gen_*.png'))
l1_gt=sorted(glob('../ccom/gt_*.png'))

gdl_psnr=[]
for i in range(10):
    gen = cv2.imread(gdl_gen[i])#[320:,620,:]
    gt = cv2.imread(gdl_gt[i])#[320:,620,:]
    p = psnr(gen, gt)
    gdl_psnr.append(p)

l1_psnr=[]
for i in range(10):
    gen = cv2.imread(l1_gen[i])#[320:,620,:]
    gt = cv2.imread(l1_gt[i])#[320:,620,:]
    p = psnr(gen, gt)
    l1_psnr.append(p)
diff = []
for i in range(10):
    d = gdl_psnr[i] - l1_psnr[i]
    diff.append(d)

print l1_psnr
print gdl_psnr
print diff
import matplotlib.pyplot as plt 
t=[1,2,3,4,5,6,7,8,9,10]
plt.figure()
plt.plot(t, gdl_psnr, "g-", label="MGAN", linewidth=2)
plt.plot(t, l1_psnr, "r--", label="GAN", linewidth=2)
#plt.plot(t, gdl_psnr, "g-", linewidth=2)
#plt.plot(t, l1_psnr, "r--", linewidth=2)

plt.ylim(0,30)
plt.xlabel("The predicted typhoon cloud image")
plt.ylabel("PSNR")
plt.grid(True)
plt.legend()
plt.savefig('psnr.png')
plt.show()
