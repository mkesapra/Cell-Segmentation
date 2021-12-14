#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:54:52 2021

@author: kesaprm
"""


from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib.lines import Line2D
import segPlots
import math


# read_file = pd.read_csv(r'M2_morphNmotility.txt')
# read_file.to_csv(r'M2_morphNmotility.csv',index=None)

df0 = pd.read_csv("Round_spinning.txt") 
df0['imageName'] = 'Round'
df1 = pd.read_csv("Multipolar_wandering.txt") 
df1['imageName'] = 'Multipolar'
df2 = pd.read_csv("Elongated_Bidirectional_Stretching.txt") 
df2['imageName'] = 'Elongated'


#Append time-series speed values for M0
df0_speed = pd.read_csv("Round_Speed_allCells.txt")
df0_speed_arr = df0_speed.to_numpy()
df0['Speed_allCells'] =  df0_speed_arr.tolist()

#Append time-series persistence values for M0
df0_per = pd.read_csv("Round_Persistence_cellWise.txt")
df0_per_arr = df0_per.to_numpy()
df0['Per_allCells'] =  df0_per_arr.tolist()



#Append time-series speed values for M0
df1_speed = pd.read_csv("Pro_Speed_allCells.txt")
df1_speed_arr = df1_speed.to_numpy()
df1['Speed_allCells'] =  df1_speed_arr.tolist()

#Append time-series persistence values for M0
df1_per = pd.read_csv("Pro_Persistence_cellWise.txt")
df1_per_arr = df1_per.to_numpy()
df1['Per_allCells'] =  df1_per_arr.tolist()


#Append time-series speed values for M0
df2_speed = pd.read_csv("Elon_Speed_allCells.txt")
df2_speed_arr = df2_speed.to_numpy()
df2['Speed_allCells'] =  df2_speed_arr.tolist()

#Append time-series persistence values for M0
df2_per = pd.read_csv("Elon_Persistence_cellWise.txt")
df2_per_arr = df2_per.to_numpy()
df2['Per_allCells'] =  df2_per_arr.tolist()



x = ['M0', 'M1', 'M2']
ecc = np.array([df0.ecc[0],df1.ecc[0],df2.ecc[0]])
sol = np.array([df0.sol[0],df1.sol[0],df2.sol[0]])
comp = np.array([df0.cmp[0],df1.cmp[0],df2.cmp[0]])

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, ecc, width=0.8, label='Ecc', color='gold', bottom=sol+comp)
plt.bar(x_pos, sol, width=0.8, label='Sol', color='silver', bottom=comp)
plt.bar(x_pos, comp, width=0.8, label='Cmp', color='#CD853F')

plt.xticks(x_pos, x)
plt.ylabel("Morphological measures")
plt.xlabel("Images")
plt.legend(loc="upper right")

plt.show()

N = 3
eccs = (df0.ecc[0], df1.ecc[0], df2.ecc[0])
sols = (df0.sol[0],df1.sol[0],df2.sol[0])
comps = (df0.cmp[0],df1.cmp[0],df2.cmp[0])

ind = np.arange(N) 
width = 0.15  
     
# plt.bar(ind,eccs, width, label='Eccentricity' )
# plt.bar(ind + width, sols, width,
#     label='Solidity')
# plt.bar((ind + 2*width), comps, width, label='Compactness')



plt.bar(ind, comps, width, label='Compactness', color='gold')
plt.bar(ind + width, eccs, width, label='Eccentricity', color='silver')
plt.bar((ind + 2*width), sols, width,
    label='Solidity', color='#CD853F')


plt.ylabel("Morphological measures",fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.xticks(ind + width / 2, ('M0','M1','M2'),fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.legend(loc='best')
plt.show()







x = range(0,240)

plt.plot(x,df0.Speed_allCells[0], c = 'r')
plt.plot(x,df1.Speed_allCells[0], c = 'g')
plt.plot(x,df2.Speed_allCells[0], c = 'b')
plt.xlim(120,180)


plt.plot(x,df0.Per_allCells[0], c = 'r')
plt.plot(x,df1.Per_allCells[0], c = 'g')
plt.plot(x,df2.Per_allCells[0], c = 'b')
plt.xlim(120,180)
plt.ylim(0,0.2)

##GPR speed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

y0_spd = df0.Speed_allCells[0]
y1_spd = df1.Speed_allCells[0]
y2_spd = df2.Speed_allCells[0]

num_frames = len(df0_per.columns)

x = np.atleast_2d(range(1,num_frames+1)).T
dy0 =2 + 1.0 * np.random.random(np.array(y0_spd).shape)
dy1 =2 + 1.0 * np.random.random(np.array(y1_spd).shape)
dy2 = 2+ 1.0 * np.random.random(np.array(y2_spd).shape)

# Instantiate a Gaussian Process model -- Kernel parameters are estimated using maximum likelihood principle.
kernel = C(1.0, (1e-3, 1e3)) * RBF(36, (1e-2, 1e2))
gp0 = GaussianProcessRegressor(kernel=kernel, alpha=dy0 ** 2,
                              n_restarts_optimizer=10)
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=dy1 ** 2,
                              n_restarts_optimizer=10)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=dy2 ** 2,
                              n_restarts_optimizer=10)

xx = np.atleast_2d(np.linspace(1, 240, 240)).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp0.fit(x,y0_spd)
gp1.fit(x,y1_spd)
gp2.fit(x,y2_spd)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred0, sigma0 = gp0.predict(xx, return_std=True)
y_pred1, sigma1 = gp1.predict(xx, return_std=True)
y_pred2, sigma2 = gp2.predict(xx, return_std=True)


plt.figure()
plt.plot(xx, y0_spd, 'r:', linewidth=2,label='M0')
plt.plot(xx, y1_spd, 'g:', linewidth=2,label='M1')
plt.plot(xx, y2_spd, 'b:', linewidth=2,label='M2')

#plt.errorbar(x, y0_spd, dy0, linestyle='',fmt='k.', markersize=5, label='Observations')
# plt.errorbar(x, y1_spd, dy1, fmt='g.', markersize=10, label='Observations')
# plt.errorbar(x, y2_spd, dy2, fmt='b.', markersize=10, label='Observations')
# plt.errorbar(x, y3_spd, dy3, fmt='m.', markersize=10, label='Observations')

#plt.plot(x, y3_spd, 'r.', markersize=10, label='Observations')
plt.plot(xx, y_pred0, 'r-')
plt.plot(xx, y_pred1, 'g-')
plt.plot(xx, y_pred2, 'b-')

#95% confidence interval.
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred0 - 1.9600 * sigma0,
                        (y_pred0 + 1.9600 * sigma0)[::-1]]),
         alpha=.5, fc='r', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred1 - 1.9600 * sigma1,
                        (y_pred1 + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='g', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred2 - 1.9600 * sigma2,
                        (y_pred2 + 1.9600 * sigma2)[::-1]]),
         alpha=.5, fc='b', ec='None')
plt.xlabel('Time in frames(2hr-3hr)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Speed (px/frames)',fontweight="bold",fontSize="12",fontname="Times New Roman")
#plt.title('Clusterwise comparision of Cell Speed',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(120,180)
#plt.ylim(5,12)
plt.legend()

#### GPR for persistence
y0_pers = df0.Per_allCells[0]
y1_pers = df1.Per_allCells[0]
y2_pers = df2.Per_allCells[0]

x = np.atleast_2d(range(1,num_frames+1)).T
dy0 = 0.01  * np.random.random(np.array(y0_pers).shape)
dy1 = 0.01  * np.random.random(np.array(y1_pers).shape)
dy2 = 0.01 * np.random.random(np.array(y2_pers).shape)

# Instantiate a Gaussian Process model -- Kernel parameters are estimated using maximum likelihood principle.
kernel = C(1.0, (1e-3, 1e3)) * RBF(36, (1e-2, 1e2))
gp0 = GaussianProcessRegressor(kernel=kernel, alpha=dy0 ** 1,
                              n_restarts_optimizer=10)
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=dy1 ** 1,
                              n_restarts_optimizer=10)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=dy2 ** 1,
                              n_restarts_optimizer=10)

xx = np.atleast_2d(np.linspace(1, 240, 240)).T
# Fit to data using Maximum Likelihood Estimation of the parameters
gp0.fit(x,y0_pers)
gp1.fit(x,y1_pers)
gp2.fit(x,y2_pers)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred0, sigma0 = gp0.predict(xx, return_std=True)
y_pred1, sigma1 = gp1.predict(xx, return_std=True)
y_pred2, sigma2 = gp2.predict(xx, return_std=True)


plt.figure()
plt.plot(xx, y0_pers, 'r:', linewidth=2,label='M0')
plt.plot(xx, y1_pers, 'g:', linewidth=2,label='M1')
plt.plot(xx, y2_pers, 'b:', linewidth=2,label='M2')

#plt.errorbar(x, y0_spd, dy0, linestyle='',fmt='k.', markersize=5, label='Observations')
# plt.errorbar(x, y1_spd, dy1, fmt='g.', markersize=10, label='Observations')
# plt.errorbar(x, y2_spd, dy2, fmt='b.', markersize=10, label='Observations')
# plt.errorbar(x, y3_spd, dy3, fmt='m.', markersize=10, label='Observations')

#plt.plot(x, y3_spd, 'r.', markersize=10, label='Observations')
plt.plot(xx, y_pred0, 'r-')
plt.plot(xx, y_pred1, 'g-')
plt.plot(xx, y_pred2, 'b-')

#95% confidence interval.
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred0 - 1.9600 * sigma0,
                        (y_pred0 + 1.9600 * sigma0)[::-1]]),
         alpha=.5, fc='r', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred1 - 1.9600 * sigma1,
                        (y_pred1 + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='g', ec='None')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_pred2 - 1.9600 * sigma2,
                        (y_pred2 + 1.9600 * sigma2)[::-1]]),
         alpha=.5, fc='b', ec='None')
plt.xlabel('Time in frames(2hr-3hr)',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Persistence',fontweight="bold",fontSize="12",fontname="Times New Roman")
#plt.title('Clusterwise comparision of Cell Persistence',fontweight="bold",fontSize="14",fontname="Times New Roman")
plt.axis('tight')
plt.xlim(120,180)
plt.ylim(0,0.2)
plt.legend()

import seaborn as sns

M2_x = pd.read_csv("B_CellCx.txt") 
M2_y =  pd.read_csv("B_CellCy.txt") 

M1_x = pd.read_csv("W_CellCx.txt") 
M1_y =  pd.read_csv("W_CellCy.txt") 

M0_x = pd.read_csv("S_CellCx.txt") 
M0_y =  pd.read_csv("S_CellCy.txt") 

from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm












################ trajectories ########################
M0_x['M0-x'] = M0_x['Var1']
M0_x['M0-y'] = M0_y['Var1']
M0_x['M1-x'] = M1_x['Var1']
M0_x['M1-y'] = M1_y['Var1']
M0_x['M2-x'] = M2_x['Var1']
M0_x['M2-y'] = M2_y['Var1']
M0_x['time'] = list(range(1, 242))
M0_x['stamp'][0:60] = 'First'
M0_x['stamp'][60:120] = 'Second'
M0_x['stamp'][120:180] = 'Third'
M0_x['stamp'][180:] = 'Fourth'

autocorrelation_plot(M0_x['M0x'], color="r",label='M0')
autocorrelation_plot(M0_x['M1-x'], color="g",label='M1')
autocorrelation_plot(M0_x['M2-x'], color="b",label='M2')
plt.grid(False)
plt.xlabel('Lag',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Autocorrelation(x)',fontweight="bold",fontSize="12",fontname="Times New Roman")


autocorrelation_plot(M0_x['M0y'], color="r",label='M0')
autocorrelation_plot(M0_x['M1-y'], color="g",label='M1')
autocorrelation_plot(M0_x['M2-y'], color="b",label='M2')
plt.grid(False)
plt.xlabel('Lag',fontweight="bold",fontSize="12",fontname="Times New Roman")
plt.ylabel('Autocorrelation(y)',fontweight="bold",fontSize="12",fontname="Times New Roman")

plt.acorr(M0_x['M0x'], maxlags = 200) 
plt.acorr(M0_x['M1-x'], maxlags = 200) 
plt.acorr(M0_x['M2-x'], maxlags = 200) 


###########Autocorrelation should be made on tuples
#tuples (x,y)
traj =[]
traj.append([M0_x['M0x'].tolist(),M0_x['M0y'].tolist()])


tupxy = [];
for k in range(len(traj[0][0])):
    tupxy.append((traj[0][0][k],traj[0][1][k]))

np.corrcoef(M0_x['M2-x'].tolist(), M0_x['M2-y'].tolist())       

dw = [];

for k in range(0, len(M0_x['Var1'])):
    x = M0_x['M0-x'].iloc[k]
    dw.append(x)

    

dw = np.zeros(len(M0_x['M0-x']))# np.array(range(0,len(M0_x['M0-x'])));
dw_num =  np.zeros(len(M0_x['M0-x']))
dw_den =  np.zeros(len(M0_x['M0-x']))


for k in range(2, len(M0_x['M0-x'])):
        dw_num[k] = dw_num[k] + (M0_x['M0-x'][k] - M0_x['M0-x'][k-1])**2

for k in range(1, len(M0_x['M0-x'])):
        dw_den[k] = dw_den[k] + M0_x['M0-x'][k]**2

dw[k] = dw_num[k]/dw_den[k]
print(dw[k])


from statsmodels.stats.stattools import durbin_watson
durbin_watson(M0_x['M0y'])


from statsmodels.formula.api import ols

#fit multiple linear regression model
model = ols('M0x  ~  M0x+M0y', data=M0_x).fit()

#view model summary
print(model.summary())
durbin_watson(model.resid)



dw_num = np.array(range(0,cluster_size)); dw_den = np.array(range(0,cluster_size)); dw =np.array(range(0,cluster_size));
for i in range(0, cluster_size):
    for k in range(2, len(cluster_y.iloc[i])):
        dw_num[i] = dw_num[i] + (cluster_y.iloc[i][k] - cluster_y.iloc[i][k-1])**2

    for k in range(1, len(cluster_y.iloc[i])):
        dw_den[i] = dw_den[i] + cluster_x.iloc[i][k]**2

    dw[i] = dw_num[i]/dw_den[i]
    print(dw[i])


########## 
rel_x = []; rel_y =[];
for i in range(0 ,len(tupxy)-1): 
    rel_x.append(M0_x['M0x'][i] - M0_x['M0x'][0])
    rel_y.append(M0_x['M0y'][i] - M0_x['M0y'][0])

rel_x1 = []; rel_y1 =[];
for i in range(0 ,len(tupxy)-1): 
    rel_x1.append(M0_x['M1-x'][i] - M0_x['M1-x'][0])
    rel_y1.append(M0_x['M1-y'][i] - M0_x['M1-y'][0])

rel_x2 = []; rel_y2 =[];
for i in range(0 ,len(tupxy)-1): 
    rel_x2.append(M0_x['M2-x'][i] - M0_x['M2-x'][0])
    rel_y2.append(M0_x['M2-y'][i] - M0_x['M2-y'][0])



plt.plot(rel_x,rel_y, 'ro-')
plt.plot(rel_x1,rel_y1, 'go-')
plt.plot(rel_x2,rel_y2, 'bo-')





######################################
palette = sns.color_palette(None, 4)

palette = ['lightcyan','lightblue','b','navy']

plt.plot(df['M0-x'][:60],df['M0-y'][:60],color = 'r')
plt.plot(df['M0-x'][60:120],df['M0-y'][60:120],color = 'y')
plt.plot(df['M0-x'][120:180],df['M0-y'][120:180],color = 'g')
plt.plot(df['M0-x'][180:],df['M0-y'][180:],color = 'b')

plt.plot(df['M1-x'][:60],df['M1-y'][:60],color = 'r')
plt.plot(df['M1-x'][60:120],df['M1-y'][60:120],color = 'y')
plt.plot(df['M1-x'][120:180],df['M1-y'][120:180],color = 'g')
plt.plot(df['M1-x'][180:],df['M1-y'][180:],color = 'b')

plt.plot(df['M2-x'][:60],df['M2-y'][:60],color = 'r')
plt.plot(df['M2-x'][60:120],df['M2-y'][60:120],color = 'y')
plt.plot(df['M2-x'][120:180],df['M2-y'][120:180],color = 'g')
plt.plot(df['M2-x'][180:],df['M2-y'][180:],color = 'b')










#######################

sns.lineplot(data=df, x="M0-x", y="M0-y", hue="stamp", palette=palette)
sns.lineplot(data=df, x="M1-x", y="M1-y", hue="stamp", palette=palette)
sns.lineplot(data=df, x="M2-x", y="M2-y", hue="stamp", palette=palette)
sns.legend_.set_title(None)

sns.relplot(data=df, x="M0-x", y="M0-y", hue="stamp", palette=palette,  kind="line")
sns.relplot(data=df, x="M1-x", y="M1-y", hue="stamp", palette=palette,  kind="line")
sns.relplot(data=df, x="M2-x", y="M2-y", hue="stamp", palette=palette,  kind="line")




import matplotlib.animation as animation
import colorsys
N = 241
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)




