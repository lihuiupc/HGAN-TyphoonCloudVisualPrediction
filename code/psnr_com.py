#####calcute the psnr of all generations and theit gts in one training process
import tensorflow as tf
import numpy as np
import os
from glob import glob
import constants as c
from scipy.misc import imsave
from pywt import dwt2,idwt2
import cv2
import math

def psnr(target, ref):
    import cv2
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255 / rmse)

img_path = '../Save/Images/localt13/Tests/'
img_step_path = sorted(glob(os.path.join(img_path, '*')))

l1_gen=sorted(glob('../ccom/gen_*.png'))
l1_gt=sorted(glob('../ccom/gt_*.png'))

for i in range(len(img_step_path)):
    img_step_path_0 = glob(os.path.join(img_step_path[i], '0/'))
    gen = sorted(glob(os.path.join(img_step_path_0[0], 'gen_*.png')))
    gt = sorted(glob(os.path.join(img_step_path_0[0], 'gt_*.png')))
    print 'path', img_step_path[i]
    for j in range(10):
        gen_img = cv2.imread(gen[j])[320:430,620:760,:]
        gt_img = cv2.imread(gt[j])[320:430,620:760,:]
        p = psnr(gen_img,gt_img)

        ccom_gen = cv2.imread(l1_gen[j])[320:430,620:760,:]
        ccom_gt = cv2.imread(l1_gt[j])[320:430,620:760,:]
        q = psnr(ccom_gen,ccom_gt)
        diff = p - q
        #print 'recursion', j, 'localt14' , p
        #print 'recursion', j, 'com' , q
        print 'recursion', j, 'psnr_diff' , diff
