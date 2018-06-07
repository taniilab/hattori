# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import pandas as pd
import numpy as np
import sys
import time

path = 'C:/Users/Hattori/Documents/2017_11_12_touhoku/current.csv'
#path = 'C:/Users/Hattori/Documents/Andor Solis/test.csv'

counter = 0
num_sim = 30
starttime = time.time()
elapsed_time = 0
# t is index.
v = []
t = []
df = []
v_set = []
counter = 0
for i in range(0, num_sim):
    v_set.append([])
    
print('start')

df = pd.read_csv(path, delimiter='\t', skiprows=[0, 1])
#df = pd.read_csv(path, delimiter=',', skiprows=[0, 1])
print(df)
df = df.replace('0', np.nan)

"""
# test
v_set[counter] = df.ix[:, 1]
for i in range(2, 4):
    print("kashikomaaaaaaa!!!!!!!!!")
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df['col2'].count(), i]], ignore_index=True)
    print(v_set[counter])
counter += 1
"""

# FI curve
v_set[counter] = df.ix[:df["Section[1] (pA)"].count(), 1]
for i in range(2, 21):
    text = "Section[" + str(i) + "] (pA)"
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df[text].count(), i]], ignore_index=True)
    print(v_set[counter])
    print(df.ix[:df[text].count(), i])
counter += 1

# state
v_set[counter] = df.ix[:df["Section[22] (pA)"].count(), 22]
for i in range(23, 42):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# V-clamp
v_set[counter] = df.ix[:df["Section[43] (pA)"].count(), 43]
for i in range(44, 93):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state
v_set[counter] = df.ix[:df["Section[94] (pA)"].count(), 94]
for i in range(95, 113):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[114] (pA)"].count(), 114]
for i in range(115, 133):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[134] (pA)"].count(), 134]
for i in range(135, 153):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[154] (pA)"].count(), 154]
for i in range(155, 173):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[174] (pA)"].count(), 174]
for i in range(175, 193):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state
v_set[counter] = df.ix[:df["Section[194] (pA)"].count(), 194]
for i in range(195, 197):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[198] (pA)"].count(), 198]
for i in range(199, 217):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[218] (pA)"].count(), 218]
for i in range(219, 237):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[238] (pA)"].count(), 238]
for i in range(239, 257):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[258] (pA)"].count(), 258]
for i in range(259, 277):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state(2 cells)
v_set[counter] = df.ix[:df["Section[278] (pA)"].count(), 278]
for i in range(279, 297):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state(single cell)
v_set[counter] = df.ix[:df["Section[298] (pA)"].count(), 298]
for i in range(299, 307):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# V-clamp
v_set[counter] = df.ix[:df["Section[308] (pA)"].count(), 308]
for i in range(309, 316):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[317] (pA)"].count(), 317]
for i in range(318, 336):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state
v_set[counter] = df.ix[:df["Section[337] (pA)"].count(), 337]
for i in range(338, 351):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[352] (pA)"].count(), 352]
for i in range(353, 371):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# state
v_set[counter] = df.ix[:df["Section[372] (pA)"].count(), 372]
for i in range(373, 376):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# V-clamp
v_set[counter] = df.ix[:df["Section[377] (pA)"].count(), 377]
for i in range(378, 390):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[391] (pA)"].count(), 391]
for i in range(392, 410):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[411] (pA)"].count(), 411]
for i in range(412, 430):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[431] (pA)"].count(), 431]
for i in range(432, 450):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[451] (pA)"].count(), 451]
for i in range(452, 470):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[471] (pA)"].count(), 471]
for i in range(472, 490):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[491] (pA)"].count(), 491]
for i in range(492, 510):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1
print("kanopero")
# FI curve
v_set[counter] = df.ix[:df["Section[511] (pA)"].count(), 511]
for i in range(512, 530):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# FI curve
v_set[counter] = df.ix[:df["Section[531] (pA)"].count(), 531]
for i in range(532, 553):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)
counter += 1

# V-clamp
v_set[counter] = df.ix[:df["Section[554] (pA)"].count(), 554]
for i in range(555, 560):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), i]], ignore_index=True)

for i in range(0, num_sim):
    t = np.arange(0, len(v_set[i]), 1)
    df = pd.DataFrame({'index': t, 'V[mV]': v_set[i]})
    df.to_csv('C:/Users/Hattori/Documents/2017_11_12_touhoku/' + str(i) + '_current.csv')
    print(str(i) + "steps")

elapsed_time = time.time() - starttime
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
