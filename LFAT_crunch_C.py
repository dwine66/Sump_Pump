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
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a',header=0, names=['X_Value','Laser 1','Laser 2', 'Laser 3','Comments']))
    return vname

def boxcar(bc_width,Input_Frame,tb,dt,Title):
    bc = bc_width
    plt.figure(Title,figsize=(8,10))
    BC_df = pd.DataFrame(index=Input_Frame.index, columns=[tb,dt])
    BC_df[tb] = Input_Frame[tb]
    BC_df[dt] = Input_Frame[dt]
    plt.suptitle(filename +  '-' + dt + ": "+ str(bc) + " pt rolling statistics" )
    roll_avg = BC_df[dt].rolling(bc).mean()
    roll_std = BC_df[dt].rolling(bc).std()
    roll_max = BC_df[dt].rolling(bc).max()
    roll_min = BC_df[dt].rolling(bc).min()
    
    roll_range = round((roll_max-roll_min).mean(),3)
    roll_mean = roll_avg.mean()
    
    print (roll_mean, roll_range)
    
    plt.subplot(311)
    plt.plot(BC_df[tb],roll_avg, 'm.',ms=2)
    plt.plot(BC_df[tb],roll_max, 'k.',ms=1)
    plt.plot(BC_df[tb],roll_min, 'k.',ms=1)
    plt.title('Mean with max/min.  Range: '+str(roll_range))
    plt.xlabel('time (sec.)')
    plt.ylabel('disp (mm)')
    
    plt.subplot(312)
    plt.plot(BC_df[tb],roll_std, 'm.',ms=1)
    plt.title('st. dev')
    plt.ylabel('disp (mm)')
    plt.xlabel('time (sec.)')
    
    plt.subplot(313)
    plt.plot(BC_df[tb],roll_avg-roll_mean, 'm.',ms=2)
    plt.show()
    return()
    
### Constants
Sample_Rate = 10000
f_LFAT = 25

### Variables
T_LFAT = 1/f_LFAT
pts_cycle=Sample_Rate/f_LFAT

### Main Code
#WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
WKdir='C:\\Users\\dwine\\Desktop\\'
#WKdir='C:\\Users\\Dave\Google Drive\\'
#WKdir="U:\\Programs\\Projects\\DSF\\DST\\20190115 LFAT runs\\"
os.chdir(WKdir)

# Read in LFAT pump data
#filename='4 slat no compensators 1500 RPM_nh.txt'
filename='motor only chocked_nh.txt'
filename='1Khz lasers only_nh.txt'
filename='laser swapped 1 with 2_nh.txt'
filename='laser 1-3 swap decoupled_nh.txt'
filename='laser 1-3 swap decoupled_nh.txt'
filename='4 slat no compensators 1500 RPM_nh_mod.txt'
LFAT_df = readcsv(filename)
print (filename,' read OK')

#Define test regimes
gap = 1
SF = 1

# Test profile times(min)
T1min = 1
R1min = 0.33
FSmin = 2
R2min = .33
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
T1C=int(T1S+(T1F-T1S)/2)
FSC=int(FSS+(FSF-FSS)/2)
T2C=int(T2C+(T2F+T2S)/2)

t_col = 'X_Value'
print ('Sample Rate (Hz): ',1/(LFAT_df[t_col][2]-LFAT_df[t_col][1]))

# Full Run Plot
C1 = 'Laser 1'
C2 = 'Laser 2'
C3 = 'Laser 3'

plt.figure('Raw Data',figsize=(9,9))
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
boxcar(1000,LFAT_df,'X_Value','Laser 1','Baseplate Boxcar')
boxcar(1000,LFAT_df,'X_Value','Laser 2','Slat 1A Boxcar')
boxcar(1000,LFAT_df,'X_Value','Laser 3','Slat 1B Boxcar')

d_col = C2
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
plt.figure('Short-Term Noise',figsize=(8,10))
plt.suptitle(filename +  ' ' + d_col + ': short-term noise')

plt.subplot(311)
plt.scatter(T1C_df[t_col],T1C_df[d_col],s=1,c='y')
plt.title('Tombstone 1')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')

plt.subplot(312)
plt.scatter(FSC_df[t_col],FSC_df[d_col],s=1,c='b')
plt.title('Full Speed')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')

plt.subplot(313)
plt.scatter(T2C_df[t_col],T2C_df[d_col],s=1,c='y')
plt.title('Tombstone 2')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')

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

print('T1 mean, 1s, range for channel',d_col,': ', T1_m, T1_s,T1_r)
print('FS mean, 1s, range for channel',d_col,': ', FS_m, FS_s,FS_r)
print('T2 mean, 1s, range for channel',d_col,': ', T2_m, T2_s,T2_r)

### Boxcar Averages

boxcar(1000,FS_df,'X_Value','Laser 2','Tombstone 1 Boxcar')

### FFT Analysis
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/

FFT_tb = FS_df
t_R1 = str(FFT_tb['X_Value'].iloc[0])
t_R2 = str(FFT_tb['X_Value'].iloc[-1])
FFT_C = FSC
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
Ca = C1
Cb = C2

FS_2C_df = LFAT_df[[t_col,Ca,Cb]]
FS_2C_df['diff']= FS_2C_df[Ca]-FS_2C_df[Cb]

plt.figure('2-channel comparisions',figsize=(9,9))
plt.suptitle(filename +  ': ' + Ca+'-'+Cb + " displacement delta" )
plt.subplot(411)
plt.plot(FS_2C_df[t_col],FS_2C_df[Ca], 'r.',ms=1)

plt.subplot(412)
plt.plot(FS_2C_df[t_col],FS_2C_df[Cb], 'b.',ms=1)

plt.subplot(413)
plt.plot(FS_2C_df[t_col],FS_2C_df['diff'], 'm.',ms=1)
plt.show()

plt.subplot(414)
plt.plot(FS_2C_df[t_col],FS_2C_df[Ca], 'r.',ms=1)
plt.plot(FS_2C_df[t_col],FS_2C_df[Cb], 'b.',ms=1)
plt.show()

boxcar(1000,FS_2C_df,'X_Value','diff', 'Full Speed Slat Delta')
### 3-channel comparisons

#C1 = 'Voltage_0'
#C2 = 'Voltage_1'
#C3 = 'Voltage_2'
xl = FSC_df['X_Value'].iloc[0]
xh = FSC_df['X_Value'].iloc[-1]
Ncyc = 5
FS_3C_df = LFAT_df[FSC-Ncyc*int(Sample_Rate/f_LFAT):FSC+Ncyc*int(Sample_Rate/f_LFAT)]
FS_3C_df['diff12']= FS_3C_df[C1]-FS_3C_df[C2]
FS_3C_df['diff13']= FS_3C_df[C1]-FS_3C_df[C3]
FS_3C_df['diff23']= FS_3C_df[C2]-FS_3C_df[C3]

Cfig = plt.figure(8,figsize=(11,8))
#Cfig.subplots(sharex=True)
#Cfig.xlim = ([xl,xh])
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

# All 3 on one axis w/bc averaging and time shift
bc2 = 10
All_r = LFAT_df.rolling(bc2).mean()
All_r['v_shift'] = All_r[C1].shift(80)
All_r['diff'] = All_r['v_shift']-All_r[C2]
plt.figure('Raw Data - all 3 on one axis',figsize=(9,6))
plt.scatter(All_r[t_col],All_r[C1],s=1,c='r')
plt.plot(All_r[t_col],All_r['v_shift'],'r-')
plt.scatter(All_r[t_col],All_r[C2],s=1,c='b')
plt.scatter(All_r[t_col],All_r[C3],s=1,c='c')

plt.scatter(All_r[t_col],All_r['diff'],s=1,c='m')

plt.title(filename +  ': full run')
plt.show()