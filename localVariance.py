#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:53:14 2020

@author: kesaprm
"""



import matplotlib.pyplot as plt
from skimage import io, color, restoration, img_as_float
import numpy as np
img = io.imread("images/t001.tif",as_gray=True)
print(img.shape)

#Checkout this page for entropy and other examples
#https://scikit-image.org/docs/stable/auto_examples/

from skimage.filters.rank import entropy
from skimage.morphology import disk
entropy_img = entropy(img, disk(20))
#plt.imshow(entropy_img, cmap=plt.cm.gray)

#Once you have the entropy iamge you can apply a threshold to segment the image
#If you're not sure which threshold works fine, skimage has a way for you to check all 

"""
from skimage.filters import try_all_threshold
fig, ax = try_all_threshold(entropy_img, figsize=(10, 8), verbose=False)
plt.show()
"""

#Now let us test Otsu segmentation. 
from skimage.filters import threshold_otsu
thresh = threshold_otsu(entropy_img)   #Just gives us a threshold value. Check in variable explorer.
binary= entropy_img <=thresh  #let us generate a binary image by separating pixels below and above threshold value.
plt.imshow(binary, cmap=plt.cm.gray)
print("The percent white region is: ", (np.sum(binary == 1)*100)/(np.sum(binary == 0) + np.sum(binary == 1)))   #Print toal number of true (white) pixels
