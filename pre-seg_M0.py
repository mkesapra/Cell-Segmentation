#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Code to get the segmentation results and calculate ecc, sol, ... other params
"""
Created on Fri Jan 29 15:22:59 2021

@author: kesaprm
"""


import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import numpy as np
import segDefinitions
import segPlots
##### Read Images and convert them to grayscale
img = cv2.imread("M0.tif",0)
imgName = 'M0'

plt.imshow(img)
plt.title('input image')
cv2.imshow('Input image',img)
cv2.waitKey()

## minimum area considered for segmentation
area = 6000
gradient_for_Ms = segDefinitions.preSegmentGrad(img)

cv2.imshow('denoised image',gradient_for_Ms)
cv2.waitKey()
plt.title('denoised image')

gradient_for_M2 = segDefinitions.preSegmentGradBlur(img)
labels = segDefinitions.generateLabels(gradient_for_Ms)
df = segDefinitions.imgregionProps(img,labels,area,imgName)

df['imageName'] = imgName

#df.to_csv('For3Clustering.csv', encoding='utf-8', mode='a', index=True)

#######################Analysis after segmentation###############################
x = df['label']
y1 = df['solidity']
y2 = df['compactness']

segPlots.linePlots_MinMeanLines(x,y1,y2,imgName)

##plotting the scatter and cluster plots for eccentricity
y3 = df['eccentricity']
xlabel = "Cell label"
ylabel = "Eccentricity"
segPlots.scatterPlots(x,y3,xlabel,ylabel,imgName)

## cluster plot with 4 clusters
# string = 'e[0,0.4)= ' +'{:}'.format(y3[y3 < 0.4].count()) + ' ->'+'{:}%'.format(int(y3[y3 < 0.4].count()/y3.count()*100))
# string2 ='e[0.4,0.6)= ' + '{:}'.format(y3[np.logical_and(y3 >= 0.4,y3<0.6)].count())+ ' ->'+'{:}%'.format(int(y3[np.logical_and(y3 >= 0.4,y3<0.6)].count()/y3.count()*100))
# string3 = 'e[0.6,0.8)= ' +'{:}'.format(y3[np.logical_and(y3 >= 0.6,y3<0.8)].count())+ ' ->'+'{:}%'.format(int(y3[np.logical_and(y3 >= 0.6,y3<0.8)].count()/y3.count()*100))
# string4 = 'e[0.8,1)= ' +'{:}'.format(y3[y3 >= 0.8].count())+ ' ->'+'{:}%'.format(int(y3[y3 >= 0.8].count()/y3.count()*100))
# vhi = plt.scatter(x[y3 >= 0.8],y3[y3 >= 0.8], c='blue',label=string4)
# hi = plt.scatter(x[np.logical_and(y3 >= 0.6,y3<0.8)],y3[np.logical_and(y3 >= 0.6,y3<0.8)], c='m',label=string3)
# lo = plt.scatter(x[np.logical_and(y3 >= 0.4,y3<0.6)],y3[np.logical_and(y3 >= 0.4,y3<0.6)], c='g',label=string2)
# vlo = plt.scatter(x[y3 < 0.4],y3[y3 < 0.4], c='red',label=string)

#### cluster plot with 3 clusters
string = 'e[0,0.5)= ' +'{:}'.format(y3[y3 < 0.5].count()) + ' ->'+'{:}%'.format(int(y3[y3 < 0.5].count()/y3.count()*100))
string2 ='e[0.5,0.8)= ' + '{:}'.format(y3[np.logical_and(y3 >= 0.5,y3<0.8)].count())+ ' ->'+'{:}%'.format(int(y3[np.logical_and(y3 >= 0.5,y3<0.8)].count()/y3.count()*100))
string3 = 'e[0.8,1)= ' +'{:}'.format(y3[y3 >= 0.8].count())+ ' ->'+'{:}%'.format(int(y3[y3 >= 0.8].count()/y3.count()*100))
vhi = plt.scatter(x[y3 >= 0.8],y3[y3 >= 0.8], c='blue',label=string3)
lo = plt.scatter(x[np.logical_and(y3 >= 0.5,y3<0.8)],y3[np.logical_and(y3 >= 0.5,y3<0.8)], c='g',label=string2)
vlo = plt.scatter(x[y3 < 0.5],y3[y3 < 0.5], c='red',label=string)


plt.xlabel("Cell label")
plt.ylabel("Eccentricity");
plt.axis('tight')
plt.grid(True)
plt.title('Eccentricity of cells in '+ imgName)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
##eccentricity histograms
segPlots.histPlots(y3,ylabel,'Frequency',imgName)

##major-axis values
y4 = df['major_axis_length']
ylabel2 ="Major_axis_length"
segPlots.scatterPlots(x,y4,xlabel,ylabel2,imgName)

##ratio-axis values
df['ratiomM'] = df['minor_axis_length']/df['major_axis_length']
y5 = df['ratiomM']
ylabel3 ="Minor_axis_length/Major_axis_length"
segPlots.scatterPlots(x,y5,xlabel,ylabel3,imgName)

##Cell area
y6=df['area']
ylabel4 ="Area"
segPlots.scatterPlots(x,y6,xlabel,ylabel4,imgName)


segPlots.histPlots(y6,ylabel4,'Frequency',imgName)


###orientation
angle_in_degrees = df['orientation'] * (180/np.pi) + 90 
ylabel5 = 'Orientation in degrees'
segPlots.scatterPlots(x,angle_in_degrees,xlabel,ylabel5,imgName)

segPlots.histPlots(angle_in_degrees,ylabel5,'Frequency',imgName)


#df.to_csv('For3Clustering.csv', encoding='utf-8',  mode='a',index=False)
#df['eccentricity'].to_csv('AllCells.csv', encoding='utf-8', index=False)

#df[np.logical_and(df['eccentricity'] > 0.5 ,df['eccentricity'] < 0.85)].to_csv('filteresEcc.csv', encoding='utf-8', index=False)
