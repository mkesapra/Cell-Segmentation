#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:15:04 2021

@author: kesaprm
"""

import matplotlib.path as mpath
import matplotlib.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np


star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()

def linePlots_MinMeanLines(x,y1,y2,imgName):
    plt.plot(x, y1, color='b', label = 'Solidity',marker=star, markersize=5, linewidth = 0.5)       
    plt.plot(x, y2, color='g', label = 'Compactness',marker=circle, markersize=5, linewidth = 0.5) 

    plt.axhline(y2.min(),color="c", linestyle="--",label='min vals',linewidth=1)
    plt.text(-8,y2.min(),np.round(y2.min(),2),color="c", fontsize=7)
    plt.axhline(y2.mean(),color="r", linestyle="--",label='mean vals',linewidth=1)
    plt.text(-8,y2.mean(),np.round(y2.mean(),2),color="r", fontsize=7)
    
    plt.axhline(y1.min(),color="c", linestyle="--",linewidth=1)
    plt.text(-8,y1.min(),np.round(y1.min(),2),color="c", fontsize=7)
    plt.axhline(y1.mean(),color="r", linestyle="--",linewidth=1)
    plt.text(-8,y1.mean(),np.round(y1.mean(),2),color="r", fontsize=7)
    
    plt.axis('tight')
    plt.title('Solidity and Compactness of cells in '+ imgName+' image')
    plt.xlabel("Cell label")
    plt.ylabel("Cell measurements");
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def scatterPlots(x,y,xlabel,ylabel,imgName):
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('tight')
    plt.grid(True)
    plt.title(imgName)

def histPlots(y,xlabel,ylabel,imgName,bc = "#0504aa"): # bc - band color - an optional paramter
    n, bins, patches = plt.hist(y, bins=30, color= bc,
                            alpha=0.6, rwidth=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(imgName)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=maxfreq)
    return n

def linePlots(x,y,xlabel,ylabel,imgName,color):
    plt.plot(x,y,color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('tight')
    plt.grid(True)
    plt.title(imgName)    