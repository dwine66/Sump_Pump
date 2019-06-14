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

# File Handling
import socket
import tkinter as tk
import tkinter.filedialog

### Functions
def readcsv(fname,Header_lst):
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a',header=12, names=Header_lst))
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
    plt.suptitle(filename +  ' :' + dt + ": "+ str(bc) + " pt rolling statistics" )
    plt.subplots_adjust(hspace=0.35)

    plt.subplot(311)
    plt.title(Title)
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

def Get_File():
    tk.Tk().withdraw() # Close the root window
    in_path = tk.filedialog.askopenfilename()
    print (in_path)
    tk.Tk().destroy()
    return in_path
### Constants


#Headers = ['time (sec)','Slat_5_LE (L1)','Slat_4_MD (L2)', 'Slat_3_TE (L3)','Comments']
#Headers = ['time (sec)','Slat_5_LE (L1)','Slat_4_TE (L2)', 'Slat_4_C (L3)','Comments']

### Variables
  
### Load Datafile
machine_name = socket.gethostname()
print ('Machine Detected: '+ machine_name)
#WKdir='S:\\Dave\\QH\\BBP\\LFAT Pump\\Data\\'
if machine_name == "TURTLE":
    WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2STB Data\\'
else:
    #WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Data\\'
    #WKdir='C:\\Users\\Dave\Google Drive\\'
    WKdir="U:\\Programs\\Projects\\DSF\\DST\\LFAT-8L\\2STB\\Test Data\\"

#WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Data\\'
#os.chdir(WKdir)
# Read in LFAT pump data
#filename='20190125_4S_C_1500_NoFeet.txt'
#filename='motor only chocked_nh.txt'
#filename='1Khz lasers only_nh.txt'
#filename='laser swapped 1 with 2_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
#filename='laser 1-3 swap decoupled_nh.txt'
filename='20190516_02S_Run 1_nh.txt'
filename='20190523 Run B_nh.txt'
#filename='20190129_12S_Run 5_nh.txt'
### File Read

#https://codeyarns.com/2014/02/25/how-to-use-file-open-dialog-to-get-file-path-in-python/
filepath = Get_File()
filename = filepath[-21:]

# Read header lines
fp = open(filepath)
Header_Line=[]
for i, line in enumerate(fp):
    if i < 11:
        Header_Line.append(line.rstrip())
        print (line)
    else:
        break

fp.close()

# Parse Headers
Run_Name = Header_Line[0][5:]
Date = Header_Line[1][6:]
Machine_Name = Header_Line[2][9:]
Test_Type = Header_Line[3][9:]
RPM = float(Header_Line[4][10:])
Test_Duration = float(Header_Line[5][35:])
Laser_1_lst=Header_Line[7].split(',')
Laser_2_lst=Header_Line[8].split(',')
Laser_3_lst=Header_Line[9].split(',') 
Notes = Header_Line[10][7:]

# Load up variables
L1_x = Laser_1_lst[2]
L2_x = Laser_2_lst[2]
L3_x = Laser_3_lst[2]

L1_z = Laser_1_lst[3]
L2_z = Laser_2_lst[3]
L3_z = Laser_3_lst[3]

L1_SF = Laser_1_lst[4]
if L1_SF.isdigit():
    L1_SF=float(L1_SF)
else:
    L1_SF=1.0

L2_SF = Laser_2_lst[4]
if L2_SF.isdigit():
    L2_SF=float(L1_SF)
else:
    L2_SF=1.0

L3_SF = Laser_3_lst[4]
if L3_SF.isdigit():
    L3_SF=float(L1_SF)
else:
    L3_SF=1.0

f_LFAT = RPM/60 # Hertz  
T_LFAT = 1/f_LFAT

# Dataframe Headers
Headers = ['time (sec)',Laser_1_lst[1]+' (L1)',Laser_2_lst[1]+' (L2)',Laser_3_lst[1]+' (L3)']#'Comments']
 
# Abbreviate Headers
C0 = Headers[0]
C1 = Headers[1]
C2 = Headers[2]
C3 = Headers[3]

# Get Data using header info
LFAT_df = readcsv(filepath,Headers)
print (filename,' read OK')

# Define some names for Columns
t_col = C0 # time column
d_col = C3 # Selected column

#Define test regimes
gap = 1 # Space between regimes
SF = 1

# Test regime times(min)
T1min = 1 # Tombstone 1
R1min = 0.5 # Ramp 1
FSmin = 3.5 # Full Speed
R2min = 0.5 # Ramp 2
T2min = 1 # Tombstone 2

Sample_Rate = int(1/(LFAT_df[t_col][2]-LFAT_df[t_col][1])) # Caclulate sample rate of Keyence
print ('Measured Sample Rate (Hz): ',Sample_Rate)

pts_cycle=Sample_Rate/f_LFAT # Data points per LFAT cycle
print ('Datapoints per LFAT cycle:',pts_cycle)

# Start and finish data points for each regime
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
T2C=int(T2S+(T2F-T2S)/2)
print ('Centers:',T1C,FSC,T2C)

#Scale End
LFAT_df[C1]=LFAT_df[C1]/L1_SF
LFAT_df[C2]=LFAT_df[C2]/L2_SF
LFAT_df[C3]=LFAT_df[C3]/L3_SF

# Define a dataframe for each regime
T1MC_df = LFAT_df[T1S:T1F]
R1MC_df = LFAT_df[R1S:R1F]
FSMC_df = LFAT_df[FSS:FSF]
R2MC_df = LFAT_df[R2S:R2F]
T2MC_df = LFAT_df[T2S:T2F]

# Create data blocks for a specific column (d_col)
LFAT_SC_df = LFAT_df[[t_col,d_col]]

T1_df = LFAT_SC_df[T1S:T1F]
R1_df = LFAT_SC_df[R1S:R1F]
FS_df = LFAT_SC_df[FSS:FSF]
R2_df = LFAT_SC_df[R2S:R2F]
T2_df = LFAT_SC_df[T2S:T2F]

# Create two-cycle short-term blocks at center of each part
T1C_df = T1_df[int((T1S-T1C)-(Sample_Rate/f_LFAT)):int((T1S-T1C)+(Sample_Rate/f_LFAT))]
FSC_df = FS_df[int((FSC-FSS)-(Sample_Rate/f_LFAT)):int((FSC-FSS)+(Sample_Rate/f_LFAT))]
T2C_df = T2_df[int((T2S-T2C)-(Sample_Rate/f_LFAT)):int((T2S-T2C)+(Sample_Rate/f_LFAT))]

# Create dataframes for short-term differential data
# This throws a warning - fix (use .loc)
Ncyc = 5 # Number of cycles to display on either side of center point of run
FS_3C_df = LFAT_df[FSC-Ncyc*int(Sample_Rate/f_LFAT):FSC+Ncyc*int(Sample_Rate/f_LFAT)]

diff12 = C1+'-'+C2
diff13 = C1+'-'+C3
diff23 = C2+'-'+C3

FS_3C_df[diff12]= FS_3C_df[C1]-FS_3C_df[C2]
FS_3C_df[diff13]= FS_3C_df[C1]-FS_3C_df[C3]
FS_3C_df[diff23]= FS_3C_df[C2]-FS_3C_df[C3]

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
plt.subplots_adjust(hspace = .35)

plt.subplot(311)
plt.scatter(LFAT_df[t_col],LFAT_df[C1],s=1,c='r')
plt.title(filename +  ' ' + C1 + ': full run', fontsize=10)
plt.ylabel('disp (mm)')

plt.subplot(312)
plt.scatter(LFAT_df[t_col],LFAT_df[C2],s=1,c='b')
plt.title(filename +  ' ' + C2 + ': full run', fontsize=10)
plt.ylabel('disp (mm)')

plt.subplot(313)
plt.scatter(LFAT_df[t_col],LFAT_df[C3],s=1,c='g')
plt.title(filename +  ' ' + C3 + ': full run', fontsize=10)
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

### ST Absolute Plots
boxcar(100,FS_3C_df,C0,C1,'Boxcar - Absolute ST ' + C1)
boxcar(100,FS_3C_df,C0,C2,'Boxcar - Absolute ST ' + C2)
boxcar(100,FS_3C_df,C0,C3,'Boxcar - Absolute ST ' + C3)

### Differentials
boxcar(100,FS_3C_df,C0, diff12,'Boxcar - ST diff12')
boxcar(100,FS_3C_df,C0, diff13,'Boxcar - ST diff13')
boxcar(100,FS_3C_df,C0, diff23,'Boxcar - ST diff23')

#boxcar(100,FS_3C_df,C0,'mdiff12','Boxcar - mdiff12')
#boxcar(100,FS_3C_df,C0,'mdiff13','Boxcar - mdiff13')
#boxcar(100,FS_3C_df,C0,'mdiff23','Boxcar - mdiff23')

### Single-channel stats
# Full run
boxcar(100,LFAT_df,C0,C1,C1 +' Boxcar - full run')
boxcar(100,LFAT_df,C0,C2,C2 +' Boxcar - full run')
boxcar(100,LFAT_df,C0,C3,C3 +' Boxcar - full run')

# Full speed - entire regime
boxcar(100,FSMC_df,C0,C1,C1+' Boxcar - Full Speed')
boxcar(100,FSMC_df,C0,C2,C2+' Boxcar - Full Speed')
boxcar(100,FSMC_df,C0,C3,C3+' Boxcar - Full Speed')

### Selected column stats
# Full Run for selected column
plt.figure(d_col+': Full Run',figsize=(6,4))
plt.scatter(LFAT_df[t_col],LFAT_df[d_col],s=1,c='m')
plt.title(filename +  ' ' + d_col + ': full run')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

# Full Speed Only
plt.figure(d_col+': Full Speed Only',figsize=(6,4))
plt.scatter(FS_df[t_col],FS_df[d_col],s=1,c='m')
plt.title(filename +  ' ' + d_col + ': full speed')
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')
plt.show()

# Ramps and Tombstones
plt.figure(d_col+': Ramps and Tombstones',figsize=(9,6))
plt.suptitle(filename + ' ' + d_col + ': ramps and tombstones')
plt.subplots_adjust(hspace=0.35)

plt.subplot(221)
plt.scatter(T1_df[t_col],T1_df[d_col],s=1,c='c')
plt.title('Tombstone 1',fontsize=10)
plt.ylabel('disp (mm)')

plt.subplot(222)
plt.scatter(R1_df[t_col],R1_df[d_col],s=1,c='y')
plt.title('Ramp 1 (up)',fontsize=10)

plt.subplot(223)
plt.scatter(R2_df[t_col],R2_df[d_col],s=1,c='y')
plt.title('Ramp 2 (down)',fontsize=10)
plt.xlabel('time (sec.)')
plt.ylabel('disp (mm)')

plt.subplot(224)
plt.scatter(T2_df[t_col],T2_df[d_col],s=1,c='c')
plt.title('Tombstone 2',fontsize=10)
plt.xlabel('time (sec.)')

plt.show()

# Short-term noise plots
plt.figure(d_col+'- Short-Term Noise',figsize=(9,6))
plt.suptitle(filename +  ' ' + d_col + '- short-term noise')
plt.subplots_adjust(hspace=0.35)

plt.subplot(311)
plt.scatter(T1C_df[t_col],T1C_df[d_col],s=1,c='c')
plt.title('Tombstone 1',fontsize=10)
plt.ylabel('disp (mm)')

plt.subplot(312)
plt.scatter(FSC_df[t_col],FSC_df[d_col],s=1,c='m')
plt.title('Full Speed',fontsize=10)
plt.ylabel('disp (mm)')

plt.subplot(313)
plt.scatter(T2C_df[t_col],T2C_df[d_col],s=1,c='c')
plt.title('Tombstone 2',fontsize=10)
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
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
fig.title('FFT of Full Speed data')
fig.suptitle(filename + ' '+ d_col + ' ' + t_R1+' to '+t_R2 + ' sec: FFT' )
ax.plot(fftfreq[i], 10 * np.log10(disp_psd[i]))
ax.set_xlim(0, 10*f_LFAT)
ax.set_xticks(np.arange(0, 10*f_LFAT, step=f_LFAT))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
fig.show()

### 2-channel comparisons
Ca = C2
Cb = C3

FS_2C_df = FSMC_df[[t_col,Ca,Cb]]
FS_2C_df['diff']= FS_2C_df[Ca]-FS_2C_df[Cb]

plt.figure('2-channel comparison',figsize=(9,10))
plt.suptitle(filename + ' ' + Ca+'-'+Cb + " Overlay" )
plt.subplots_adjust(hspace=0.35,wspace=0.25)

plt.subplot(411)
plt.title(Ca,fontsize=10)
plt.plot(FS_2C_df[t_col],FS_2C_df[Ca], 'r.',ms=1)

plt.subplot(412)
plt.title(Cb,fontsize=10)
plt.plot(FS_2C_df[t_col],FS_2C_df[Cb], 'b.',ms=1)

plt.subplot(413)
plt.title(Ca+' - '+Cb,fontsize=10)
plt.plot(FS_2C_df[t_col],FS_2C_df['diff'], 'm.',ms=1)
plt.show()

plt.subplot(414)
plt.title('Overlay',fontsize=10)
plt.plot(FS_2C_df[t_col],FS_2C_df[Ca], 'r.',ms=1)
plt.plot(FS_2C_df[t_col],FS_2C_df[Cb], 'b.',ms=1)
#plt.tight_layout(pad=1.0,h_pad=1.0)
plt.show()

### 3-channel comparisons

xl = FSC_df[C0].iloc[0]
xh = FSC_df[C0].iloc[-1]
Cfig = plt.figure(8,figsize=(11,8))
#Cfig.subplots(sharex=True)
#Cfig.xlim = ([xl,xh])
Cfig.suptitle(filename +  '- displacement deltas' )
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
a4.plot(FS_3C_df[t_col],FS_3C_df[diff12], 'm.',ms=1)
a4.set_title('1-2')

a5 = plt.subplot(337)
a5.plot(FS_3C_df[t_col],FS_3C_df[diff13], '.',color='brown',ms=1)
a5.set_title('1-3')

a6 = plt.subplot(338)
a6.plot(FS_3C_df[t_col],FS_3C_df[diff23], 'y.',ms=1)
a6.set_title('2-3')

Cfig.show()

### Boxcar w/timeshift
bc2 = 10
data_shift=80
C_shift=C1
C_diff=C2
All_r = FS_3C_df.rolling(bc2).mean()
# Shift C1 by data_shift points
All_r['v_shift'] = All_r[C_shift].shift(data_shift)
# Subtract C2 from it
All_r['diff'] = All_r['v_shift']-All_r[C_diff]
plt.figure('2ch: 10pt Boxcar time shift',figsize=(9,6))
plt.scatter(All_r[t_col],All_r[C1],s=1,c='r')
plt.plot(All_r[t_col],All_r['v_shift'],'r',marker=None)
plt.scatter(All_r[t_col],All_r[C2],s=1,c='b')
plt.scatter(All_r[t_col],All_r[C3],s=1,c='g')

plt.scatter(All_r[t_col],All_r['diff'],s=1,c='m')

plt.title(filename +  '10pt Boxcar: '+str(C_shift)+' vs '+ str(C_diff)+' shifted by '+str(data_shift)+' points',fontsize=10)
plt.show()