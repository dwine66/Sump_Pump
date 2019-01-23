# -*- coding: utf-8 -*-

# Platting data

### Libraries
import csv
import numpy as np
import pandas as pd
import os
import re
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime, timedelta
import dateutil

### Functions
def readcsv(fname):
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a',header=0))
    return vname

### Constants

Sample_Rate = 3000
f_LFAT = 25

### Variables
T_LFAT = 1/f_LFAT
pts_cycle=Sample_Rate/f_LFAT

### Main Code
WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
WKdir='C:\\Users\\Dave\\Desktop\\'
WKdir="U:\\Programs\\Projects\\DSF\\DST\\20190115 LFAT runs\\"
os.chdir(WKdir)

# Read in LFAT pump data
filename='20190114 LFAT-2L_1500_nh.txt'

LFAT_df = readcsv(filename)
print (filename,' read OK')
#Define test regimes
gap = 1
SF = 0.1
T1S = 1
T1F = int(900000*SF)
R1S = T1F+gap
R1F = int(R1S+180000*SF)
FSS = R1F+gap
FSF = int(FSS+2800000*SF)
R2S = FSS+gap
R2F = int(R2S+180000*SF)
T2S = R2F+gap
T2F = int(T2S + 900000*SF)

T1C=int((T1F+T1S)/2)
FSC=int((FSF+FSS)/2)
T2C=int((T2F+T2S)/2)

T1_df = LFAT_df[T1S:T1F]
R1_df = LFAT_df[R1S:R1F]
FS_df = LFAT_df[FSS:FSF]
R2_df = LFAT_df[R2S:R2F]
T2_df = LFAT_df[T2S:T2F]

T1C_df = T1_df[int(T1C-(Sample_Rate/f_LFAT)):int(T1C+(Sample_Rate/f_LFAT))]
FSC_df = FS_df[int(FSC-(Sample_Rate/f_LFAT)):int(FSC+(Sample_Rate/f_LFAT))]
T2C_df = T2_df[int(T2C-(Sample_Rate/f_LFAT)):int(T2C+(Sample_Rate/f_LFAT))]

##LFAT_df.drop('entry_id',axis=1,inplace = True)
##Timestamp = pd.to_datetime(LFAT_df['created_at'])
#LFAT_df.set_index('created_at',inplace = True)
#LFAT_df.index.name='Timestamp'
#LFAT_df.columns = ['Alert Level','Milone Level','Float Sensor','Temp. (C)','Board V','Milone Raw','AD590 Raw', 'Unix Timestamp']
time = pd.to_numeric(LFAT_df['time(sec)'])
Disp = pd.to_numeric(LFAT_df['disp(mm)'])
#print(Timestamp[1])

## Basic Plots
# Full Run
plt.figure(1,figsize=(9,3))
plt.scatter(LFAT_df['time(sec)'],LFAT_df['disp(mm)'],s=1,c='r')
plt.title(filename + ': full run')
plt.show()

# Full Speed
plt.figure(2,figsize=(6,3))
plt.scatter(FS_df['time(sec)'],FS_df['disp(mm)'],s=1,c='b')
plt.title(filename + ': full speed')
plt.show()

#Ramps and Tombstones
plt.figure(3,figsize=(9,6))
plt.suptitle(filename + ': ramps and tombstones')

plt.subplot(221)
plt.scatter(T1_df['time(sec)'],T1_df['disp(mm)'],s=1,c='y')
plt.title('Tombstone 1')

plt.subplot(222)
plt.scatter(R1_df['time(sec)'],R1_df['disp(mm)'],s=1,c='g')
plt.title('Ramp 1')

plt.subplot(224)
plt.scatter(R2_df['time(sec)'],R2_df['disp(mm)'],s=1,c='g')
plt.title('Ramp 2')

plt.subplot(223)
plt.scatter(T2_df['time(sec)'],T2_df['disp(mm)'],s=1,c='y')
plt.title('Tombstone 2')
plt.show()

# Short-term noise plots
plt.figure(4,figsize=(9,3))
plt.suptitle(filename + ': short-term noise')

plt.subplot(131)
plt.scatter(T1C_df['time(sec)'],T1C_df['disp(mm)'],s=1,c='c')
plt.title('Tombstone 1')

plt.subplot(132)
plt.scatter(FSC_df['time(sec)'],FSC_df['disp(mm)'],s=1,c='c')
plt.title('Full Speed 1')

plt.subplot(133)
plt.scatter(T2C_df['time(sec)'],T2C_df['disp(mm)'],s=1,c='c')
plt.title('Tombstone 2')


## Short-term noise on tombstones
T1_m = T1_df['disp(mm)'].mean()
FS_m = FS_df['disp(mm)'].mean()
T2_m = T2_df['disp(mm)'].mean()
T1_s = T1_df['disp(mm)'].std()
FS_s = FS_df['disp(mm)'].std()
T2_s = T2_df['disp(mm)'].std()

print('T1 mean, 1s:', T1_m, T1_s)
print('FS mean, 1s:', FS_m, FS_s)
print('T2 mean, 1s:', T2_m, T2_s)
### Boxcar Averages

bc = 100
roll_mean = Disp.rolling(bc).mean()
plt.plot(roll_mean)

roll_std = Disp.rolling(bc).std()
plt.plot(roll_std)

# Basic Plotting
#plt.ion()
LFAT_df.plot('time(sec)','disp(mm)','scatter',style='+')

plt.plot(time,Disp,'ro',ms=1)


### FFT Analysis
disp_fft = np.fft.fft(Disp)
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real)
plt.show()


#disp_ts = pd.Series(LFAT_df['disp(mm)'],index='time(sec)')            
            #,title='LFAT Pump Water Depth (cm)')#, xlim=(len(Milone)-Toffset,len(Milone)))
#print (Toffset)
# Pull out Milone data and timestamp

## Plot
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.grid(True,which="both",ls="-", color='0.65')
ax1.plot(Milone)
plt.xlabel('Date')
plt.ylabel('Water Level (cm)')
plt.title('Water Level in LFAT vs. time')

"""
# Annotate specific data points
for i in range(0,len(ss_df.index)):
    plt.annotate(ss_df.index[i],xy=(np.array(ss_Density)[i],ss_Temp[i]),xycoords='data')