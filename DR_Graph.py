# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 04:34:14 2017

Initiated 10/22/2017

DR_Graph - plots data from CSV file of drag reduction data

Rev 0.1 1/13/2020: Initital functionality

@author: Dave
"""
### Libraries

# Data Munging
import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime, timedelta
import dateutil

# File Handling
import os
import csv
import socket
import tkinter as tk
import tkinter.filedialog

### Functions
def readcsv(fname):
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a'))
    return vname

def Get_File():
    tk.Tk().withdraw() # Close the root window
    in_path = tk.filedialog.askopenfilename()
    #print (in_path)
    tk.Tk().destroy()
    return in_path
### Constants

### Main Code
    
machine_name = socket.gethostname()
print ('Machine Detected: '+ machine_name)

if machine_name == "TURTLE":
    WKdir='C:\\Users\\Dave\\Desktop\\'
else:
    #WKdir='C:\\Users\\Dave\\Desktop\\LFAT-2L Data\\'
    #WKdir='C:\\Users\\Dave\Google Drive\\'
    WKdir="U:\\Programs\\Projects\\DSF\\DST\\"

filename = Get_File()
#filename = filepath[-len(WKdir)+1:]

# Read in DR data
#filename='DST DR Database_20200113.csv'
print (filename)
DR_df = readcsv(filename)

DR_df.set_index('Dataset',inplace = True)

# Create dataframe

DR_Type_df = DR_df[['Retau','DR%']]
DR_OC_df = DR_Type_df.filter(like='Other Work - Comp',axis=0)
DR_OE_df = DR_Type_df.filter(like='Other Work - Exp',axis=0)
DR_2LI_df = DR_Type_df.filter(like='LFAT-2L (Initial)',axis=0)
DR_2LR_df = DR_Type_df.filter(like='LFAT-2L (Rebuilt)',axis=0)
DR_8LA_df = DR_Type_df.filter(like='LFAT-8L (A only)',axis=0)
DR_UoM_LES_df = DR_Type_df.filter(like='UoM LES',axis=0)
DR_8L_ABCD_df = DR_Type_df.filter(like='LFAT-8L (ABCD) - CHWA',axis=0)
DR_8L_FE_df = DR_Type_df.filter(like='LFAT-8L (ABCD) - FE',axis=0)

#plt.scatter(DR_Type_df['Retau'],DR_Type_df['DR%'])
fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)

ax1.scatter(DR_OC_df['Retau'],DR_OC_df['DR%'],marker="x",c='k',s=15)
ax1.scatter(DR_OE_df['Retau'],DR_OE_df['DR%'],marker="+",c='k',s=15)
ax1.scatter(DR_2LI_df['Retau'],DR_2LI_df['DR%'],marker="o", c='w')
ax1.scatter(DR_2LR_df['Retau'],DR_2LR_df['DR%'],marker="o", c='y')
ax1.scatter(DR_8LA_df['Retau'],DR_8LA_df['DR%'],marker="o", c='g')
ax1.scatter(DR_UoM_LES_df['Retau'],DR_UoM_LES_df['DR%'],marker="^")
ax1.scatter(DR_8L_ABCD_df['Retau'],DR_8L_ABCD_df['DR%'],marker="s")
ax1.scatter(DR_8L_FE_df['Retau'],DR_8L_FE_df['DR%'],marker="s", c='r')

ax1.set_xlabel('Re.tau')
ax1.set_ylabel('% Drag Reduction')
ax1.set_title('Drag Reduction vs. Retau')

plt.show

### Plotting
fig = plt.figure()  # an empty figure with no axes
fig.suptitle('No axes on this figure')  # Add a title so we know which it is
fig, ax_lst = plt.subplots(1, 1)  # a figure with a 2x2 grid of Axes

