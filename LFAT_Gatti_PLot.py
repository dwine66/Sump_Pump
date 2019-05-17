# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:59:31 2019

@author: dwine
"""
### Gatti Plotter
# -*- coding: utf-8 -*-

# Platting data

### Libraries
import csv
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import os
import re
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime, timedelta
import dateutil

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm



### Functions
def readcsv(fname,Header_lst):
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a',header=0, names=Header_lst))
    return vname

def boxcar(bc_width,Input_Frame,tb,dt,Title):
    bc = bc_width

    BC_df = pd.DataFrame(index=Input_Frame.index, columns=[tb,dt])
    BC_df[tb] = Input_Frame[tb]
    BC_df[dt] = Input_Frame[dt]
    
    roll_avg = BC_df[dt].rolling(bc).mean()
    roll_std = BC_df[dt].rolling(bc).std()
    roll_max = BC_df[dt].rolling(bc).max()
    roll_min = BC_df[dt].rolling(bc).min()
    
    roll_range = round((roll_max-roll_min).mean(),3)
    roll_mean = roll_avg.mean()
    
    print (Title,' Mean:',round(roll_mean,3), 'Range:',roll_range)
    
    plt.figure(filename+'-'+ Title,figsize=(8,10))   
    plt.suptitle(filename +  '-' + dt + ": "+ str(bc) + " pt rolling statistics" )
    plt.subplots_adjust(hspace=0.35)

    plt.subplot(311)
    plt.title('Absolute data')
    plt.plot(BC_df[tb],roll_avg, 'm.',ms=1)
   #plt.plot(BC_df[tb],roll_max, 'k.',ms=1)
    #plt.plot(BC_df[tb],roll_min, 'k.',ms=1)

    #plt.xlabel('time (sec.)')
    plt.ylabel('disp (mm)')

    plt.subplot(312)
    plt.title('Stability around mean')
    plt.plot(BC_df[tb],roll_avg-roll_mean, 'm.',ms=2)
#    plt.errorbar(BC_df[tb],roll_avg-roll_mean,yerr=roll_std,'m.',ms=2)
    plt.ylim([-0.6,0.6])
    #plt.xlabel('time (sec.)')
    plt.ylabel('disp (mm)')
    plt.show()
    
    plt.subplot(313)
    plt.title('1s and Range: '+str(roll_range))
    plt.plot(BC_df[tb],roll_std, 'm.',ms=1)
    #plt.plot(BC_df[tb],roll_range, 'k.',ms=1)
    plt.ylim([0,0.6])
    plt.title('st. dev')
    plt.ylabel('disp (mm)')
    plt.xlabel('time (sec.)')

    return()
    
### Constants
f_LFAT = 25
Headers = ['time (sec)','Slat_4C (1)','Slat_12R (2)', 'Slat_12L (3)','Comments']
Headers = ['w+','k+','Drag Reduction']
#Headers = ['time (sec)','Slat_5_LE (L1)','Slat_4_TE (L2)', 'Slat_4_C (L3)','Comments']

### Variables
T_LFAT = 1/f_LFAT

### Data Load Code
#WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
WKdir='C:\\Users\\dwine\\Desktop\\LFAT-2L Data\\'
#WKdir='C:\\Users\\Dave\Google Drive\\'
WKdir="U:\\Programs\\Projects\\DSF\\DST\\Library\\2016 Gatti Map Data\\"
#WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Data\\'
os.chdir(WKdir)

# Read in LFAT pump data
filename='20190125_4S_C_1500_NoFeet.txt'
#filename='motor only chocked_nh.txt'
#filename='1Khz lasers only_nh.txt'
#filename='laser swapped 1 with 2_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
filename='20190128_4S_NC_1500_Feet.txt'
filename='Gatti_Ref3_plot.csv'

LFAT_df = readcsv(filename,Headers)
print (filename,' read OK')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

# Plot contour curves
cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)

ax.clabel(cset, fontsize=9, inline=1)

plt.show()

# from https://matplotlib.org/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

wmin = min(LFAT_df['w+'])
kmin = min(LFAT_df['k+'])
DRmin = min(LFAT_df['Drag Reduction'])

wmax = max(LFAT_df['w+'])
kmax = max(LFAT_df['k+'])
DRmax = (LFAT_df['Drag Reduction'])
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, DRmin, DRmax in [('r', 'o', -50, -25)]:
    xs = LFAT_df['w+']
    ys = LFAT_df['k+']
    zs = LFAT_df['Drag Reduction']
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('w+')
ax.set_ylabel('k+')
ax.set_zlabel('Drag Reduction')

plt.show()



