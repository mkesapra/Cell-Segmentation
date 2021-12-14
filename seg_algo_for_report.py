#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:06:51 2021

@author: kesaprm
"""

import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage import measure, io, img_as_ubyte
from skimage.color import label2rgb
import pandas as pd
import math



##### Read Images and convert them to grayscale
#img = cv2.imread("/Users/kesaprm/FY19_20/Spring2020/Project/FS-Tracer_Data/Cody_Ms/M0_w2_NO_Plate_D_p00_0_C04f00d4.PNG",0)

img = cv2.imread("M1_01.tif")
imgName = 'M0'

plt.imshow(img,cmap='gray')
plt.title('input image')
plt.axis('off')
#cv2.imshow('Input image',img)
#cv2.waitKey()


#cv2.imwrite('Grayimage.png',img)

########gradient_for_Ms = segDefinitions.preSegmentGrad(img)
equ = cv2.equalizeHist(img)


plt.hist(img.flat, bins=100, range=(0,255))
plt.xlabel('Pixel Intensity',fontweight="bold",fontname="Times New Roman",fontSize="14")
plt.ylabel('No. of pixels',fontweight="bold",fontname="Times New Roman",fontSize="14")


plt.hist(equ.flat, bins=100, range=(0,255))
plt.xlabel('Pixel Intensity',fontweight="bold",fontname="Times New Roman",fontSize="14")
plt.ylabel('No. of pixels',fontweight="bold",fontname="Times New Roman",fontSize="14")

plt.imshow(equ,cmap='gray')
plt.axis('off')










































































cv2.imwrite('ClaheImg.png',clahe_img)


#median filter
median = cv2.medianBlur(clahe_img,5)
plt.imshow(median,cmap='gray')
plt.axis('off')

bilateral = cv2.bilateralFilter(clahe_img,9,75,75)
plt.imshow(bilateral,cmap='gray')
plt.axis('off')

nlm = cv2.fastNlMeansDenoising(clahe_img,9,11,21)
plt.imshow(nlm,cmap='gray')
plt.axis('off')


filterSize =(5, 5) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  
                               filterSize) 
# Perform erosion and dilation separately and then subtract
erosion = cv2.erode(clahe_img, kernel, iterations = 5)
dilation = cv2.dilate(clahe_img, kernel, iterations=1)
gradient1 = dilation - erosion#cv2.imshow('denoised image',gradient_blur)
#cv2.waitKey()

cv2.imwrite('erosion.png',erosion)
cv2.imwrite('dilation.png',dilation)
cv2.imwrite('gradient1.png',gradient1)


plt.imshow(gradient1,cmap='gray')
plt.title('gradient1 image')
plt.axis('off')

######gradient_for_M2 = segDefinitions.preSegmentGradBlur(img)
######labels = segDefinitions.generateLabels(gradient_for_Ms)

kernel_convolve = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(clahe_img,-1,kernel_convolve)

plt.imshow(dst,cmap='gray')
plt.title('dst image')

# ##fill holes in grayscale image
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# res = cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel)

# plt.imshow(res)

# closing = cv2.morphologyEx(clahe_img, cv2.MORPH_CLOSE, kernel)
# opening = cv2.morphologyEx(clahe_img, cv2.MORPH_OPEN, kernel)
# tophat_img = cv2.morphologyEx(opening,  
#                               cv2.MORPH_BLACKHAT, 
#                               kernel) 
retBG,th =cv2.threshold(dst,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

plt.imshow(th,cmap='gray')
plt.title('thresh - median blur on gradient')

masked = cv2.bitwise_and(dst, dst, mask=th)
plt.imshow(masked)





cv2.imwrite('masked.png',masked)

## for edge detection
edges = cv2.Canny(th,100,200)
plt.imshow(edges,cmap='gray')
plt.title('edges')

cv2.imwrite('edges.png',edges)


# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
k = np.array([[0,0,1], [0,1,0], [1,0,0]])
th = ndimage.binary_fill_holes(th)
th = ndimage.convolve(th, k, mode='wrap')

plt.imshow(th, cmap='gray')
plt.axis('off')
plt.savefig('major-fillholes.png', dpi=300)

#cv2.imwrite('fillholes.png',th.astype(int))


D = ndimage.distance_transform_edt(th)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=th)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th)
   
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
plt.imshow(labels,cmap="gray")
plt.title('labels')


cv2.imwrite('labels.png',labels)


label_image = measure.label(labels)
image_label_overlay = label2rgb(label_image, image = img, bg_label=0)
plt.imshow(image_label_overlay,cmap='gray')
plt.axis('off')
plt.savefig('image_label_overlay.png', dpi=300)

## minimum area considered for segmentation
area = 8800
props = measure.regionprops_table(label_image,img, properties = ['label','area', 'convex_area','orientation',
                                             'major_axis_length',
                                             'minor_axis_length', 'perimeter','eccentricity'])

df = pd.DataFrame(props)
df = df[df['area'] > area]
#solidity = area/ convex hull area. Solidity measures the density of an object
df['solidity'] = pd.Series(df['area']/df['convex_area'], index = df.index)
df['compactness'] = pd.Series((4* 3.14* df['area'])/df['perimeter']**2, index = df.index)
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(image_label_overlay, cmap=plt.cm.gray)
regions = measure.regionprops(label_image)
# Add labels to the plot
style = dict(size=16, color= 'yellow') #Style for the text to be added to the image

#orientation float
#Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

for props in regions:
    if(props.area > area):
         y, x = props.centroid   #Gives coordinates for the object centroid
         label = props.label  #Gives the label number for each object/region
         bbox = props.bbox
         ax.text(x, y, label, **style)   #Add text to the plot at given coordinates. In this case the text is label number. 
         y0, x0 = props.centroid
         orientation = props.orientation
         #minor axis of each ellipse
         x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
         y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
         #major axis of each ellipse
         x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
         y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

         ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
         ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
         ax.plot(x0, y0, '.g', markersize=15)

         minr, minc, maxr, maxc = props.bbox
         bx = (minc, maxc, maxc, minc, minc)
         by = (minr, minr, maxr, maxr, minr)
         ax.plot(bx, by, '-b', linewidth=2.5)
                  
         
#plt.title(imgName)
plt.axis('off')
plt.savefig('major-minor.png', dpi=300)
plt.show()
