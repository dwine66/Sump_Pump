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
plt.scatter(T1C_df['time(sec)'],T1C_df['disp(mm)'],s=1,c='y')
plt.title('Tombstone 1')

plt.subplot(132)
plt.scatter(FSC_df['time(sec)'],FSC_df['disp(mm)'],s=1,c='b')
plt.title('Full Speed')

plt.subplot(133)
plt.scatter(T2C_df['time(sec)'],T2C_df['disp(mm)'],s=1,c='y')
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

### FFT Analysis
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
FSC2_df = FS_df[int(FSC-(10*Sample_Rate/f_LFAT)):int(10*FSC+(Sample_Rate/f_LFAT))]

time = pd.to_numeric(FSC2_df['time(sec)'])
Disp = pd.to_numeric(FSC2_df['disp(mm)'])
disp_fft = sp.fftpack.fft(Disp)
disp_psd = np.abs(disp_fft)**2

fftfreq = sp.fftpack.fftfreq(len(disp_psd),1/Sample_Rate)
i = fftfreq > 0
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(disp_psd[i]))
ax.set_xlim(0, 100)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')


### Boxcar Averages
bc = 1000
plt.figure(5,figsize=(10,4))
plt.suptitle(filename + ": "+ str(bc) + " pt rolling statistics" )
roll_mean = T1_df['disp(mm)'].rolling(bc).mean()
roll_std = T1_df['disp(mm)'].rolling(bc).std()
roll_max = T1_df['disp(mm)'].rolling(bc).max()
roll_min = T1_df['disp(mm)'].rolling(bc).min()
plt.subplot(121)
plt.plot(T1_df['time(sec)'],roll_mean,'m.',ms=1)
plt.plot(T1_df['time(sec)'],roll_max, 'm.',ms=1)
plt.plot(T1_df['time(sec)'],roll_min, 'm.',ms=1)
plt.title('Mean')
plt.subplot(122)
plt.plot(T1_df['time(sec)'],roll_std, 'm.',ms=1)

plt.title('st. dev')

# Basic Plotting
#plt.ion()
LFAT_df.plot('time(sec)','disp(mm)','scatter',style='+')

plt.plot(time,Disp,'ro',ms=1)
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

