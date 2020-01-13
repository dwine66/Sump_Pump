# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 04:34:14 2017

Initiated 10/22/2017

DR_Graph - plots data from CSV file of drag reduction data

Rev 0.1 1/13/2020: Initital functionality

@author: Dave
"""
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
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a'))
    return vname

### Constants

### Main Code
WKdir='C:\\Users\\Dave\\Desktop\\'
os.chdir(WKdir)

# Read in DR data
filename='DST DR Database_20200113.csv'
print (filename)
DR_df = readcsv(filename)

DR_df.set_index('Dataset',inplace = True)

# Create dataframe

DR_Type_df = DR_df[['Retau','DR%']]

plt.scatter(DR_Type_df['Retau'],DR_Type_df['DR%'])

### Plotting
fig = plt.figure()  # an empty figure with no axes
fig.suptitle('No axes on this figure')  # Add a title so we know which it is

fig, ax_lst = plt.subplots(1, 1)  # a figure with a 2x2 grid of Axes






Sump_df.columns = ['Alert Level','Milone Level','Float Sensor','Temp. (C)','Board V','Milone Raw','AD590 Raw', 'Unix Timestamp']
Milone = pd.to_numeric(Sump_df['Milone Level'])

Sump_ts=pd.Series(Sump_df['Milone Level'],index=Timestamp)
#pd.to_datetime(Milone.index)

DR_df.plot('DR%')

Sump_avg = Milone.rolling(50).mean()
plt.plot(Sump_avg)

Sump_ts.groupby(lambda x: x.month).mean()
Sump_ts.groupby(lambda x: x.week).mean()

# Use pandas plot function
Toffset=300
Milone.plot(title='Sump Pump Water Depth (cm)', xlim=(len(Milone)-Toffset,len(Milone)))
print (Toffset)
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
plt.title('Water Level in sump vs. time')

"""
# Annotate specific data points
for i in range(0,len(ss_df.index)):
    plt.annotate(ss_df.index[i],xy=(np.array(ss_Density)[i],ss_Temp[i]),xycoords='data')
"""