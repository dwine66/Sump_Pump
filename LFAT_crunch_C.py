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
WKdir='C:\\Users\\Dave\Google Drive\\'
WKdir="U:\\Programs\\Projects\\DSF\\DST\\20190115 LFAT runs\\"
os.chdir(WKdir)

# Read in LFAT pump data
filename='20190114 LFAT-2L_1500_nh.txt'

LFAT_df = readcsv(filename)
print (filename,' read OK')
#Define test regimes
gap = 1
SF = 0.1

# Test profile times
T1min = 5
T2min = 5
R1min = 1
R2min = 1
FSmin = 10

T1S = 1
T1F = T1S + int(T1min*60*Sample_Rate*SF)
R1S = T1F + gap
R1F = R1S + int(R1min*60*Sample_Rate*SF)
FSS = R1F + gap
FSF = FSS + int(FSmin*60*Sample_Rate*SF)
R2S = FSF + gap
R2F = R2S + int(R2min*60*Sample_Rate*SF)
T2S = R2F + gap
T2F = T2S + int(T2min*60*Sample_Rate*SF)

print(T1S,T1F,R1S,R1F,FSS,FSF,R2S,R2F,T2S,T2F)

# Need to subtract out absolute reference otherwise indexing goes beyond end of dataframe
T1C=int((T1F+T1S)/2)-T1S
FSC=int((FSF+FSS)/2)-FSS
T2C=int((T2F+T2S)/2)-T2S
print (T1C,FSC,T2C)

t_col = 'time(sec)'
d_col = 'disp(mm)'

LFAT_SC_df = LFAT_df[[t_col,d_col]]
# Single-channel stats
T1_df = LFAT_SC_df[T1S:T1F]
R1_df = LFAT_SC_df[R1S:R1F]
FS_df = LFAT_SC_df[FSS:FSF]
R2_df = LFAT_SC_df[R2S:R2F]
T2_df = LFAT_SC_df[T2S:T2F]

T1C_df = T1_df[int(T1C-(Sample_Rate/f_LFAT)):int(T1C+(Sample_Rate/f_LFAT))]
FSC_df = FS_df[int(FSC-(Sample_Rate/f_LFAT)):int(FSC+(Sample_Rate/f_LFAT))]
T2C_df = T2_df[int(T2C-(Sample_Rate/f_LFAT)):int(T2C+(Sample_Rate/f_LFAT))]

## Basic Plots
# Full Run
plt.figure(1,figsize=(9,3))
plt.scatter(LFAT_df[t_col],LFAT_df[d_col],s=1,c='r')
plt.title(filename +  ' ' + d_col + ': full run')
plt.show()

# Full Speed
plt.figure(2,figsize=(6,3))
plt.scatter(FS_df[t_col],FS_df[d_col],s=1,c='b')
plt.title(filename +  ' ' + d_col + ': full speed')
plt.show()

#Ramps and Tombstones
plt.figure(3,figsize=(9,6))
plt.suptitle(filename + ' ' + d_col + ': ramps and tombstones')

plt.subplot(221)
plt.scatter(T1_df[t_col],T1_df[d_col],s=1,c='y')
plt.title('Tombstone 1')

plt.subplot(222)
plt.scatter(R1_df[t_col],R1_df[d_col],s=1,c='g')
plt.title('Ramp 1')

plt.subplot(223)
plt.scatter(T2_df[t_col],T2_df[d_col],s=1,c='y')
plt.title('Tombstone 2')

plt.subplot(224)
plt.scatter(R2_df[t_col],R2_df[d_col],s=1,c='g')
plt.title('Ramp 2')
plt.show()

# Short-term noise plots
plt.figure(4,figsize=(9,4))
plt.suptitle(filename +  ' ' + d_col + ': short-term noise')

plt.subplot(131)
plt.scatter(T1C_df[t_col],T1C_df[d_col],s=1,c='y')
plt.title('Tombstone 1')

plt.subplot(132)
plt.scatter(FSC_df[t_col],FSC_df[d_col],s=1,c='b')
plt.title('Full Speed')

plt.subplot(133)
plt.scatter(T2C_df[t_col],T2C_df[d_col],s=1,c='y')
plt.title('Tombstone 2')
plt.show()

## Short-term noise on tombstones
T1_m = T1_df[d_col].mean()
FS_m = FS_df[d_col].mean()
T2_m = T2_df[d_col].mean()
T1_s = T1_df[d_col].std()
FS_s = FS_df[d_col].std()
T2_s = T2_df[d_col].std()

print('T1 mean, 1s:', T1_m, T1_s)
print('FS mean, 1s:', FS_m, FS_s)
print('T2 mean, 1s:', T2_m, T2_s)

### Boxcar Averages
bc = 1000
plt.figure(5,figsize=(10,4))
plt.suptitle(filename +  ' ' + d_col + ": "+ str(bc) + " pt rolling statistics" )
roll_avg = T1_df[d_col].rolling(bc).mean()
roll_std = T1_df[d_col].rolling(bc).std()
roll_max = T1_df[d_col].rolling(bc).max()
roll_min = T1_df[d_col].rolling(bc).min()
plt.subplot(121)
plt.plot(T1_df[t_col],roll_avg, 'm.',ms=1)
plt.plot(T1_df[t_col],roll_max, 'k.',ms=1)
plt.plot(T1_df[t_col],roll_min, 'k.',ms=1)
plt.title('Mean')
plt.subplot(122)
plt.plot(T1_df[t_col],roll_std, 'm.',ms=1)
plt.title('st. dev')
plt.show()

### FFT Analysis
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
FSC2_df = FS_df[int(FSC-(10*Sample_Rate/f_LFAT)):int(10*FSC+(Sample_Rate/f_LFAT))]

time = pd.to_numeric(FSC2_df[t_col])
Disp = pd.to_numeric(FSC2_df[d_col])
disp_fft = sp.fftpack.fft(Disp)
disp_psd = np.abs(disp_fft)**2

fftfreq = sp.fftpack.fftfreq(len(disp_psd),1/Sample_Rate)
i = fftfreq > 0
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(disp_psd[i]))
ax.set_xlim(0, 100)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
fig.show()


### Multi-channel comparisons

C1 = 'disp(mm)'
C2 = 'disp(mm)'

FS_MC_df = LFAT_df[[t_col,C1,C2]]
FS_MC_diff = FS_MC_df[C1]-FS_MC_df[C2]
FS_MC_diff[t_col]=FS_MC_df[t_col]
FS_MC_diff = FS_MC_diff[tcol,C1,C2]

