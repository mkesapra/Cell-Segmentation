#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  31 19:31:36 2020

@author: kesaprm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/kesaprm/Learning/frame1.jpg',0)
img2 = cv2.imread('images/20X MQAE 7.5mM well 1 + IVM t1.PNG',0)
eq_img = cv2.equalizeHist(img)
clahe =cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

# sub = np.abs(img-img2)
cl_img2 =clahe.apply(img)
# ret,threshSub =cv2.threshold(cl_img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(cl_img2)


plt.hist(cl_img2.flat,bins=100, range=(0,40))
plt.title('After BG using Running average algorithm for MQAE Well1')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
ret,thresh =cv2.threshold(cl_img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("Equalized image,",eq_img)
cv2.imshow("OTSU,",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
