''' Intellectual Ventures CONFIDENTIAL-SUBJECT TO NDA
This document contains confidential information that shall be distributed, routed 
or made available solely in accordance with a nondisclosure agreement (NDA).  
Copyright © 2019 Intellectual Ventures Management, LLC (IV).  All rights reserved.
'''
# Drag Reduction Landscape Plot
# Initiated 12/17/2019 DWW

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

def Get_File():
    tk.Tk().withdraw() # Close the root window
    in_path = tk.filedialog.askopenfilename()
    print (in_path)
    tk.Tk().destroy()
    return in_path

### Constants

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

### File Read

#https://codeyarns.com/2014/02/25/how-to-use-file-open-dialog-to-get-file-path-in-python/
filepath = Get_File()
filename = filepath[-21:]
# DST Drag Reduction Landscape_20191217.csv
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
Run_Name = Header_Line[0][4:]
Date = Header_Line[1][5:]
Machine_Name = Header_Line[2][8:]
Test_Type = Header_Line[3][10:]
RPM = float(Header_Line[4][9:])
FS_Duration = float(Header_Line[5][34:])
Laser_1_lst=Header_Line[7].split(',')
Laser_2_lst=Header_Line[8].split(',')
Laser_3_lst=Header_Line[9].split(',') 
Notes = Header_Line[10][6:]

# Load up variables
L1_x = Laser_1_lst[2]
L2_x = Laser_2_lst[2]
L3_x = Laser_3_lst[2]

L1_z = Laser_1_lst[3]
L2_z = Laser_2_lst[3]
L3_z = Laser_3_lst[3]


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
SF = 1 # Sample Rate scaling factor

# Test regime times(min)
T1min = 1 # Tombstone 1
R1min = 0.5 # Ramp 1
FSmin = FS_Duration # Full Speed
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


### Basic Plots
# Raw Data
plt.figure('Raw Data',figsize=(8,10))
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





