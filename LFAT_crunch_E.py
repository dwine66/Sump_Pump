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
f_LFAT = 12.5 # in Hertz
Headers = ['time (sec)','Slat_BR (1)','Slat_BL (2)', 'Slat_AC (3)','Comments']
#Headers = ['time (sec)','Slat_5_LE (L1)','Slat_4_MD (L2)', 'Slat_3_TE (L3)','Comments']
#Headers = ['time (sec)','Slat_5_LE (L1)','Slat_4_TE (L2)', 'Slat_4_C (L3)','Comments']

### Variables
T_LFAT = 1/f_LFAT

### Data Load Code
#WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
#WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2STB Data\\'
#WKdir='C:\\Users\\Dave\Google Drive\\'
WKdir="U:\\Programs\\Projects\\DSF\\DST\\LFAT-8L\\2STB\\Test Data\\"
#WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Data\\'
os.chdir(WKdir)

# Read in LFAT pump data
#filename='20190125_4S_C_1500_NoFeet.txt'
#filename='motor only chocked_nh.txt'
#filename='1Khz lasers only_nh.txt'
#filename='laser swapped 1 with 2_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
#filename='20190128_4S_NC_1500_Feet.txt'
filename='20190516_02S_Run 1_nh.txt'
filename='20190521 Run C_nh.txt'

LFAT_df = readcsv(filename,Headers)
print (filename,' read OK')

C0 = Headers[0]
C1 = Headers[1]
C2 = Headers[2]
C3 = Headers[3]

t_col = C0
d_col = C2

#Define test regimes
gap = 1
SF = 1

# Test profile times(min)
T1min = 1
R1min = .5
FSmin = 3
R2min = .5
T2min = 1

print ('Measured Sample Rate (Hz): ',int(1/(LFAT_df[t_col][2]-LFAT_df[t_col][1])))
Sample_Rate = int(1/(LFAT_df[t_col][2]-LFAT_df[t_col][1]))
pts_cycle=Sample_Rate/f_LFAT
print ('Datapoints per LFAT cycle:',pts_cycle)
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

# Need to subtract out absolute reference otherwise indexing goes beyond end of dataframe

T1C=int(T1S+(T1F-T1S)/2)
FSC=int(FSS+(FSF-FSS)/2)
T2C=int(T2S+(T2F+T2S)/2)
print ('Centers:',T1C,FSC,T2C)

T1MC_df = LFAT_df[T1S:T1F]
R1MC_df = LFAT_df[R1S:R1F]
FSMC_df = LFAT_df[FSS:FSF]
R2MC_df = LFAT_df[R2S:R2F]
T2MC_df = LFAT_df[T2S:T2F]

# Create data blocks for each part of run
LFAT_SC_df = LFAT_df[[t_col,d_col]]

T1_df = LFAT_SC_df[T1S:T1F]
R1_df = LFAT_SC_df[R1S:R1F]
FS_df = LFAT_SC_df[FSS:FSF]
R2_df = LFAT_SC_df[R2S:R2F]
T2_df = LFAT_SC_df[T2S:T2F]

# Create two-cycle short-term blocks  at center of each part
T1C_df = T1_df[int((T1S-T1C)-(Sample_Rate/f_LFAT)):int((T1S-T1C)+(Sample_Rate/f_LFAT))]
FSC_df = FS_df[int((FSC-FSS)-(Sample_Rate/f_LFAT)):int((FSC-FSS)+(Sample_Rate/f_LFAT))]
T2C_df = T2_df[int((T2S-T2C)-(Sample_Rate/f_LFAT)):int((T2S-T2C)+(Sample_Rate/f_LFAT))]

Ncyc = 5 # Number of cycles to display
FS_3C_df = LFAT_df[FSC-Ncyc*int(Sample_Rate/f_LFAT):FSC+Ncyc*int(Sample_Rate/f_LFAT)]
FS_3C_df['diff12']= FS_3C_df[C1]-FS_3C_df[C2]
FS_3C_df['diff13']= FS_3C_df[C1]-FS_3C_df[C3]
FS_3C_df['diff23']= FS_3C_df[C2]-FS_3C_df[C3]

#FS_3C_C1m = FS_3C_df[C1].mean()
#FS_3C_C2m = FS_3C_df[C2].mean()
#FS_3C_C3m = FS_3C_df[C3].mean()
#
#FS_3C_df['mdiff12']= (FS_3C_df[C1]-FS_3C_C1m) - (FS_3C_df[C2]-FS_3C_C2m)
#FS_3C_df['mdiff13']= (FS_3C_df[C1]-FS_3C_C1m) - (FS_3C_df[C1]-FS_3C_C3m)
#FS_3C_df['mdiff23']= (FS_3C_df[C2]-FS_3C_C2m) - (FS_3C_df[C2]-FS_3C_C3m)


### Basic Plots
# Raw Data
plt.figure('Raw Data',figsize=(9,9))
plt.suptitle(filename + ': Raw Data')
plt.subplots_adjust(hspace = .25)

plt.subplot(311)
plt.scatter(LFAT_df[t_col],LFAT_df[C1],s=1,c='r')
plt.title(filename +  ' ' + C1 + ': full run')
plt.ylabel('disp (mm)')

plt.subplot(312)
plt.scatter(LFAT_df[t_col],LFAT_df[C2],s=1,c='r')
plt.title(filename +  ' ' + C2 + ': full run')
plt.ylabel('disp (mm)')

plt.subplot(313)
plt.scatter(LFAT_df[t_col],LFAT_df[C3],s=1,c='r')
plt.title(filename +  ' ' + C3 + ': full run')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

### Differentials

boxcar(100,FS_3C_df,C0,C1,'Boxcar - ' + C1)
boxcar(100,FS_3C_df,C0,C2,'Boxcar - ' + C2)
boxcar(100,FS_3C_df,C0,C3,'Boxcar - ' + C3)

boxcar(100,FS_3C_df,C0,'diff12','Boxcar - diff12')
boxcar(100,FS_3C_df,C0,'diff13','Boxcar - diff13')
boxcar(100,FS_3C_df,C0,'diff23','Boxcar - diff23')

#boxcar(100,FS_3C_df,C0,'mdiff12','Boxcar - mdiff12')
#boxcar(100,FS_3C_df,C0,'mdiff13','Boxcar - mdiff13')
#boxcar(100,FS_3C_df,C0,'mdiff23','Boxcar - mdiff23')

# Single-channel stats
# Full run
boxcar(100,LFAT_df,C0,C1,C1+' Boxcar - full run')
boxcar(100,LFAT_df,C0,C2,C2+' Boxcar - full run')
boxcar(100,LFAT_df,C0,C3,C3+' Boxcar - full run')
# Full speed
boxcar(100,FSMC_df,C0,C1,C1+'Baseplate Boxcar')
boxcar(100,FSMC_df,C0,C2,C2+'Slat 1A Boxcar')
boxcar(100,FSMC_df,C0,C3,C3+'Slat 1B Boxcar')

# Full Run
plt.figure(1,figsize=(9,3))
plt.scatter(LFAT_df[t_col],LFAT_df[d_col],s=1,c='r')
plt.title(filename +  ' ' + d_col + ': full run')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

# Full Speed Only
plt.figure(2,figsize=(6,4))
plt.scatter(FS_df[t_col],FS_df[d_col],s=1,c='b')
plt.title(filename +  ' ' + d_col + ': full speed')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

#Ramps and Tombstones
plt.figure(3,figsize=(9,6))
plt.suptitle(filename + ' ' + d_col + ': ramps and tombstones')
plt.subplots_adjust(hspace=0.25)

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

### FFT Analysis
# https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/

FFT_tb = FS_df
t_R1 = str(FFT_tb[C0].iloc[0])
t_R2 = str(FFT_tb[C0].iloc[-1])
FFT_C = FSC-FSS
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

### 3-channel comparisons

xl = FSC_df[C0].iloc[0]
xh = FSC_df[C0].iloc[-1]
Cfig = plt.figure(8,figsize=(11,8))
#Cfig.subplots(sharex=True)
#Cfig.xlim = ([xl,xh])
Cfig.suptitle(filename +  ': displacement deltas' )
Cfig.subplots_adjust(hspace=0.35,wspace=0.25)

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