#  -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import pandas as pd
import numpy as np
import sys
import time

#path = 'C:/Users/Hattori/Documents/Andor Solis/13_20div_rhn.csv'
# path = 'C:/Users/Hattori/Documents/Andor Solis/test.csv'
path = "Z:/Box Sync/Personal/Documents/touhoku_patch/20180501_cortex/"

#mode = "current"
#unit = "(pA)"
mode = "voltage"
unit = "(mV)"

num_sim = 100
counter = 0
starttime = time.time()
elapsed_time = 0
#  t is index.
v = []
t = []
df = []
v_set = []
counter = 0
for i in range(0, num_sim):
    v_set.append([])
    
print('start')
df = pd.read_csv(path + mode + ".csv", delimiter='\t', skiprows=[0, 1])
# df = pd.read_csv(path + "voltage.csv", delimiter=',', skiprows=[0, 1])
#df = df.replace('0', np.nan)
df = df.replace(0, np.nan)


index = ['15', '20', '35', '40', '55', '60', '75', '80', '95', '98', '101', '104', '107', '110', '113', '116', '131', '136'] 

# Initialize
top =  1

# Create columns for each experiment content
for counter, last in enumerate(index):
    v_set[counter] = df.ix[:df["Section[" + str(top) + "] " + str(unit)].count(), "Section[" + str(top) + "] " + str(unit)]
    print(v_set[int(counter)])
    for j in range(top+1, int(last)):
        v_set[int(counter)] = pd.concat([v_set[int(counter)], df.ix[:df["Section[" + str(j) + "] " + str(unit)].count(), "Section[" + str(j) + "] " + str(unit)]], ignore_index=True)
    top = int(last)

for i in range(0, counter):
    t = np.arange(0, len(v_set[i]), 1)
    df = pd.DataFrame({"index": t, mode + unit: v_set[i]})
    df.to_csv(path + mode + "/" + mode + str(i) + '.csv')
    print(str(i) + "steps")

elapsed_time = time.time() - starttime
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
