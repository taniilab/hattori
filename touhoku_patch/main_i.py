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
path = "Z:/Box Sync/Personal/Documents/20180420_touhoku_patch/current.csv"
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
df = pd.read_csv(path, delimiter='\t', skiprows=[0, 1])
# df = pd.read_csv(path, delimiter=',', skiprows=[0, 1])
print(df)
#df = df.replace('0', np.nan)
df = df.replace(0, np.nan)
print(df)

# 0. FI curve
v_set[counter] = df.ix[:df["Section[0] (pA)"].count(), "Section[0] (pA)"]
for i in range(1, 43):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 1. state
v_set[counter] = df.ix[:df["Section[43] (pA)"].count(), "Section[43] (pA)"]
for i in range(44, 63):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 2. V-clamp
v_set[counter] = df.ix[:df["Section[63] (pA)"].count(), "Section[63] (pA)"]
for i in range(64, 68):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 3. state
v_set[counter] = df.ix[:df["Section[68] (pA)"].count(), "Section[68] (pA)"]
for i in range(69, 83):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 4. FI curve
v_set[counter] = df.ix[:df["Section[83] (pA)"].count(), "Section[83] (pA)"]
for i in range(84, 89):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 5. FI curve
v_set[counter] = df.ix[:df["Section[89] (pA)"].count(), "Section[89] (pA)"]
for i in range(90, 104):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 6. FI curve
v_set[counter] = df.ix[:df["Section[104] (pA)"].count(), "Section[104] (pA)"]
counter += 1

# 7. FI curve
v_set[counter] = df.ix[:df["Section[105] (pA)"].count(), "Section[105] (pA)"]
for i in range(106, 117):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 8. state
v_set[counter] = df.ix[:df["Section[117] (pA)"].count(), "Section[117] (pA)"]
for i in range(118, 132):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 9. FI curve
v_set[counter] = df.ix[:df["Section[132] (pA)"].count(), "Section[132] (pA)"]
for i in range(133, 137):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[137] (pA)"].count(), "Section[137] (pA)"]
for i in range(138, 152):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[152] (pA)"].count(), "Section[152] (pA)"]
for i in range(153, 157):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

    
# 10. FI curve
v_set[counter] = df.ix[:df["Section[157] (pA)"].count(), "Section[157] (pA)"]
for i in range(158, 164):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[164] (pA)"].count(), "Section[164] (pA)"]
for i in range(165, 179):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1
    
# 10. FI curve
v_set[counter] = df.ix[:df["Section[179] (pA)"].count(), "Section[179] (pA)"]
for i in range(180, 184):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[184] (pA)"].count(), "Section[184] (pA)"]
for i in range(185, 199):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[199] (pA)"].count(), "Section[199] (pA)"]
for i in range(200, 210):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[210] (pA)"].count(), "Section[210] (pA)"]
for i in range(211, 225):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1
 
# 10. FI curve
v_set[counter] = df.ix[:df["Section[225] (pA)"].count(), "Section[225] (pA)"]
for i in range(226, 234):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (pA)"].count(), "Section[" + str(i) + "] (pA)"]], ignore_index=True)
counter += 1

for i in range(0, counter):
    t = np.arange(0, len(v_set[i]), 1)
    df = pd.DataFrame({'index': t, 'I[pA]': v_set[i]})
    df.to_csv('Z:/Box Sync/Personal/Documents/20180420_touhoku_patch/current' + str(i) + '.csv')
    print(str(i) + "steps")

elapsed_time = time.time() - starttime
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
