#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:31:49 2021

@author: kesaprm
"""

import cv2
import os
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import numpy as np
import segDefinitions
import segPlots
##### Read Images and convert them to grayscale
img_path = "48h 5min/M0"

for filename in os.listdir(img_path):
    img = cv2.imread(os.path.join(img_path, filename),0)
    imgName = os.path.join(img_path, filename)
    imgName = imgName[12:17]
    ## minimum area considered for segmentation
    area = 500
    gradient_for_Ms = segDefinitions.preSegmentGrad(img)
    gradient_for_M2 = segDefinitions.preSegmentGradBlur(img)
    labels = segDefinitions.generateLabels(gradient_for_Ms)
    df = segDefinitions.imgregionProps(img,labels,area,imgName)    
    df['imageName'] = imgName


    #######################Analysis after segmentation###############################
    # x = df['label']
    # y1 = df['solidity']
    # y2 = df['compactness']
    
    # segPlots.linePlots_MinMeanLines(x,y1,y2,imgName)
    
    ##plotting the scatter and cluster plots for eccentricity
    # y3 = df['eccentricity']
    # xlabel = "Cell label"
    # ylabel = "Eccentricity"
    # segPlots.scatterPlots(x,y3,xlabel,ylabel,imgName)


    # #### cluster plot with 3 clusters
    # string = 'e[0,0.5)= ' +'{:}'.format(y3[y3 < 0.5].count()) + ' ->'+'{:}%'.format(int(y3[y3 < 0.5].count()/y3.count()*100))
    # string2 ='e[0.5,0.8)= ' + '{:}'.format(y3[np.logical_and(y3 >= 0.5,y3<0.8)].count())+ ' ->'+'{:}%'.format(int(y3[np.logical_and(y3 >= 0.5,y3<0.8)].count()/y3.count()*100))
    # string3 = 'e[0.8,1)= ' +'{:}'.format(y3[y3 >= 0.8].count())+ ' ->'+'{:}%'.format(int(y3[y3 >= 0.8].count()/y3.count()*100))
    # vhi = plt.scatter(x[y3 >= 0.8],y3[y3 >= 0.8], c='blue',label=string3)
    # lo = plt.scatter(x[np.logical_and(y3 >= 0.5,y3<0.8)],y3[np.logical_and(y3 >= 0.5,y3<0.8)], c='g',label=string2)
    # vlo = plt.scatter(x[y3 < 0.5],y3[y3 < 0.5], c='red',label=string)
    
    
    # plt.xlabel("Cell label")
    # plt.ylabel("Eccentricity");
    # plt.axis('tight')
    # plt.grid(True)
    # plt.title('Eccentricity of cells in '+ imgName)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # ##eccentricity histograms
    # segPlots.histPlots(y3,ylabel,'Frequency',imgName)

    # ##major-axis values
    # y4 = df['major_axis_length']
    # ylabel2 ="Major_axis_length"
    # segPlots.scatterPlots(x,y4,xlabel,ylabel2,imgName)
    
    # ##ratio-axis values
    # df['ratiomM'] = df['minor_axis_length']/df['major_axis_length']
    # y5 = df['ratiomM']
    # ylabel3 ="Minor_axis_length/Major_axis_length"
    # segPlots.scatterPlots(x,y5,xlabel,ylabel3,imgName)
    
    # ##Cell area
    # y6=df['area']
    # ylabel4 ="Area"
    # segPlots.scatterPlots(x,y6,xlabel,ylabel4,imgName)
    
    
    # segPlots.histPlots(y6,ylabel4,'Frequency',imgName)


    # ###orientation
    # angle_in_degrees = df['orientation'] * (180/np.pi) + 90 
    # ylabel5 = 'Orientation in degrees'
    # segPlots.scatterPlots(x,angle_in_degrees,xlabel,ylabel5,imgName)
    
    # segPlots.histPlots(angle_in_degrees,ylabel5,'Frequency',imgName)


#df.to_csv('For3Clustering.csv', encoding='utf-8',  mode='a',index=False)
#df['eccentricity'].to_csv('AllCells.csv', encoding='utf-8', index=False)

#df[np.logical_and(df['eccentricity'] > 0.5 ,df['eccentricity'] < 0.85)].to_csv('filteresEcc.csv', encoding='utf-8', index=False)
