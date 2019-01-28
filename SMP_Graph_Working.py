# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 04:34:14 2017

Initiated 10/22/2017

SMP_Graph - plots data from sump pump

Rev 0.1 10/22/2017: Initital functionality

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
from dateutil.parser import parse

### Functions
def readcsv(fname):
    vname = pd.DataFrame(pd.read_csv(fname,na_values='n/a'))
    return vname

### Constants

### Main Code
WKdir='S:\\Dave\\QH\\BBP\\Sump Pump\\Data\\'
os.chdir(WKdir)

# Read in sump pump data
filename='20180731 Sump Pump Data_TS.csv'
print (filename)
Sump_df = readcsv(filename)
HeaderList = ['Datestamp','index2','Alert','Water_Level_(cm)','Float_Sensor','Temp(C)','Board_V','Milone_Counts','AD590_Counts','Unix_Time']
Sump_df.columns = HeaderList

#pd.to_numeric(Sump_df['Milone_Level_(cm)'])

# Make this a pandas time series
pd.to_datetime(Sump_df['Datestamp'])
Sump_df.set_index(pd.DatetimeIndex(Sump_df['Datestamp']),inplace = True)

#Sump_df.index.name='Datestamp'
Sump_df.drop(columns=['index2'],inplace=True)

## Plotting

Sump_df['Water_Level_(cm)'].plot()
#
#fig=plt.figure('test',figsize=(9,3))
#plt.plot(Sump_df['Water_Level_(cm)'],'r.')
#plt.show