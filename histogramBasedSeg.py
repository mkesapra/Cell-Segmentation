#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:06:08 2020

@author: kesaprm
"""

from skimage import io
from matplotlib import pyplot as plt
import numpy as np

img = io.imread("images_Sanh/01targethmagcomp_Bottom Slide_D_p01_0_A01f10d2.JPG")
img2 = io.imread("images_Sanh/01targethmagcomp_Bottom Slide_D_p01_0_A01f15d2.JPG")

plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')  
plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')  

#Let's clean the noise using edge preserving filter.
#As mentioned in previous tutorial, my favorite is NLM

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float

float_img = img_as_float(img)
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))


denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                               patch_size=5, patch_distance=3, multichannel=True)
                           
denoise_img_as_8byte = img_as_ubyte(denoise_img)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')

#Let's look at the histogram to see howmany peaks we have. 
#Then pick the regions for our histogram segmentation.

plt.hist(denoise_img_as_8byte.flat, bins=100, range=(0,100),density=True)
plt.grid(True)
  
plt.xlabel('x-axis') 
plt.ylabel('y-axis')   #.flat returns the flattened numpy array (1D)
plt.title('Intensity histogram of A01f10d2.JPG')

float_img2 = img_as_float(img2)
sigma_est2 = np.mean(estimate_sigma(float_img2, multichannel=True))


denoise_img2 = denoise_nl_means(float_img2, h=1.15 * sigma_est2, fast_mode=False, 
                               patch_size=5, patch_distance=3, multichannel=True)
                           
denoise_img_as_8byte2 = img_as_ubyte(denoise_img2)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')

#Let's look at the histogram to see howmany peaks we have. 
#Then pick the regions for our histogram segmentation.

plt.hist(denoise_img_as_8byte2.flat, bins=100, range=(0,100),density=True) 
plt.grid(True)
  
plt.xlabel('x-axis') 
plt.ylabel('y-axis') 
plt.title('Intensity histogram of A01f15d2.JPG')