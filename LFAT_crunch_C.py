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
Sample_Rate = 10000
f_LFAT = 25

### Variables
T_LFAT = 1/f_LFAT
pts_cycle=Sample_Rate/f_LFAT

### Main Code
#WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Testing_20190125\\'
#WKdir='C:\\Users\\Dave\Google Drive\\'
#WKdir="U:\\Programs\\Projects\\DSF\\DST\\20190115 LFAT runs\\"
os.chdir(WKdir)

# Read in LFAT pump data
filename='20190125 4-slat_nh.txt'
#filename='20190124 LFAT Background Run.txt'
LFAT_df = readcsv(filename)
print (filename,' read OK')
#Define test regimes
gap = 1
SF = 1

# Test profile times(min)
T1min = 1
R1min = 0.5
FSmin = 5
R2min = 0.5
T2min = 1

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

#print(T1S,T1F,R1S,R1F,FSS,FSF,R2S,R2F,T2S,T2F)

# Need to subtract out absolute reference otherwise indexing goes beyond end of dataframe
T1C=int((T1F+T1S)/2)-T1S
FSC=int((FSF+FSS)/2)-FSS
T2C=int((T2F+T2S)/2)-T2S
#print (T1C,FSC,T2C)


t_col = 'X_Value'
print ('Sample Rate (Hz): ',1/(LFAT_df[t_col][2]-LFAT_df[t_col][1]))

# Full Run Plot
C1 = 'Voltage_0'
C2 = 'Voltage_1'
C3 = 'Voltage_2'

plt.figure(0,figsize=(9,9))
plt.suptitle(filename + ': Raw Data')
plt.subplot(311)
plt.scatter(LFAT_df[t_col],LFAT_df[C1],s=1,c='r')
plt.title(filename +  ' ' + C1 + ': full run')
plt.subplot(312)
plt.scatter(LFAT_df[t_col],LFAT_df[C2],s=1,c='r')
plt.title(filename +  ' ' + C2 + ': full run')
plt.subplot(313)
plt.scatter(LFAT_df[t_col],LFAT_df[C3],s=1,c='r')
plt.title(filename +  ' ' + C3 + ': full run')
plt.show()


# Single-channel stats

d_col = C1
#LFAT_df['dummy']=LFAT_df[d_col]+0.01
#LFAT_df['dummy2']=LFAT_df[d_col]-0.01

LFAT_SC_df = LFAT_df[[t_col,d_col]]

# Create data blocks for each part of run
T1_df = LFAT_SC_df[T1S:T1F]
R1_df = LFAT_SC_df[R1S:R1F]
FS_df = LFAT_SC_df[FSS:FSF]
R2_df = LFAT_SC_df[R2S:R2F]
T2_df = LFAT_SC_df[T2S:T2F]

# Create two-cycle short-term blocks  at center of each part
T1C_df = T1_df[int(T1C-(Sample_Rate/f_LFAT)):int(T1C+(Sample_Rate/f_LFAT))]
FSC_df = FS_df[int(FSC-(Sample_Rate/f_LFAT)):int(FSC+(Sample_Rate/f_LFAT))]
T2C_df = T2_df[int(T2C-(Sample_Rate/f_LFAT)):int(T2C+(Sample_Rate/f_LFAT))]

## Basic Plots
# Full Run
plt.figure(1,figsize=(9,3))
plt.scatter(LFAT_df[t_col],LFAT_df[d_col],s=1,c='r')
plt.title(filename +  ' ' + d_col + ': full run')
plt.show()

# Full Speed Only
plt.figure(2,figsize=(6,4))
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
T1_m = round(T1_df[d_col].mean(),3)
FS_m = round(FS_df[d_col].mean(),3)
T2_m = round(T2_df[d_col].mean(),3)

T1_s = round(T1_df[d_col].std(),3)
FS_s = round(FS_df[d_col].std(),3)
T2_s = round(T2_df[d_col].std(),3)

T1_r = round(T1_df[d_col].max()-T1_df[d_col].min(),3)
FS_r = round(FS_df[d_col].max()-FS_df[d_col].min(),3)
T2_r = round(T2_df[d_col].max()-T2_df[d_col].min(),3)

print('T1 mean, 1s, range for channel ',d_col,': ', T1_m, T1_s,T1_r)
print('FS mean, 1s, range for channel ',d_col,': ', FS_m, FS_s,FS_r)
print('T2 mean, 1s, range for channel ',d_col,': ', T2_m, T2_s,T2_r)

### Boxcar Averages
bc = 1000
plt.figure('Boxcar Averages',figsize=(8,10))
plt.suptitle(filename +  '-' + d_col + ": "+ str(bc) + " pt rolling statistics" )
roll_avg = T1_df[d_col].rolling(bc).mean()
roll_std = T1_df[d_col].rolling(bc).std()
roll_max = T1_df[d_col].rolling(bc).max()
roll_min = T1_df[d_col].rolling(bc).min()

roll_range = round((roll_max-roll_min).mean(),3)

print (roll_range)

plt.subplot(211)
plt.plot(T1_df[t_col],roll_avg, 'm.',ms=2)
plt.plot(T1_df[t_col],roll_max, 'k.',ms=1)
plt.plot(T1_df[t_col],roll_min, 'k.',ms=1)
plt.title('Mean with max/min.  Range: '+str(roll_range))
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')

plt.subplot(212)
plt.plot(T1_df[t_col],roll_std, 'm.',ms=1)
plt.title('st. dev')
plt.ylabel('disp (mm)')
plt.xlabel('time (sec.)')
plt.show()

### FFT Analysis
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/

FFT_tb = T1_df
t_R1 = str(FFT_tb['X_Value'].iloc[0])
t_R2 = str(FFT_tb['X_Value'].iloc[-1])
FFT_C = T1C
FSC2_df = FFT_tb[int(FFT_C-(10*Sample_Rate/f_LFAT)):int(10*FFT_C+(Sample_Rate/f_LFAT))]

time = pd.to_numeric(FSC2_df[t_col])
Disp = pd.to_numeric(FSC2_df[d_col])
disp_fft = sp.fftpack.fft(Disp)
disp_psd = np.abs(disp_fft)**2

fftfreq = sp.fftpack.fftfreq(len(disp_psd),1/Sample_Rate)
i = fftfreq > 0
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.suptitle(filename + ' '+ d_col + ' ' + t_R1+' to '+t_R2 + ' sec: FFT' )
ax.plot(fftfreq[i], 10 * np.log10(disp_psd[i]))
ax.set_xlim(0, 100)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
fig.show()

### 2-channel comparisons

#C1 = 'disp(mm)'
#C2 = 'dummy'
Ca = 'Voltage_1'
Cb = 'Voltage_2'

FS_2C_df = LFAT_df[[t_col,Ca,Cb]]
FS_2C_df['diff']= FS_2C_df[Ca]-FS_2C_df[Cb]

plt.figure(7,figsize=(9,9))
plt.suptitle(filename +  ': ' + Ca+'-'+Cb + " displacement delta" )
plt.subplot(311)
plt.plot(FS_2C_df[t_col],FS_2C_df[Ca], 'r.',ms=1)

plt.subplot(312)
plt.plot(FS_2C_df[t_col],FS_2C_df[Cb], 'b.',ms=1)

plt.subplot(313)
plt.plot(FS_2C_df[t_col],FS_2C_df['diff'], 'm.',ms=1)
plt.show()

### 3-channel comparisons

#C1 = 'Voltage_0'
#C2 = 'Voltage_1'
#C3 = 'Voltage_2'

FS_3C_df = LFAT_df
FS_3C_df['diff12']= FS_3C_df[C1]-FS_3C_df[C2]
FS_3C_df['diff13']= FS_3C_df[C1]-FS_3C_df[C3]
FS_3C_df['diff23']= FS_3C_df[C2]-FS_3C_df[C3]

Cfig = plt.figure(8,figsize=(9,9))
Cfig.suptitle(filename +  ': displacement deltas' )
a1 = plt.subplot(331)
a1.plot(FS_3C_df[t_col],FS_3C_df[C1], 'r.',ms=1)
a1.set_title(C1)

a2 = plt.subplot(335)
a2.plot(FS_3C_df[t_col],FS_3C_df[C2], 'b.',ms=1)
a2.set_title(C2)

a3 = plt.subplot(339)
a3.plot(FS_3C_df[t_col],FS_3C_df[C3], 'g.',ms=1)
a3.set_title(C3)

a4 = plt.subplot(334)
a4.plot(FS_3C_df[t_col],FS_3C_df['diff12'], 'm.',ms=1)
a4.set_title('1-2')

a5 = plt.subplot(337)
a5.plot(FS_3C_df[t_col],FS_3C_df['diff13'], '.',color='brown',ms=1)
a5.set_title('1-3')

a6 = plt.subplot(338)
a6.plot(FS_3C_df[t_col],FS_3C_df['diff23'], 'y.',ms=1)
a6.set_title('2-3')

Cfig.show()
