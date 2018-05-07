#  -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import pandas as pd
import numpy as np
import sys
import time

path = 'C:/Users/Hattori/Documents/Andor Solis/13_20div_rhn.csv'
# path = 'C:/Users/Hattori/Documents/Andor Solis/test.csv'

counter = 0
num_sim = 30
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
df = df.replace('0', np.nan)

"""
#  test
v_set[counter] = df.ix[:, 1]
for i in range(2, 4):
    print("kashikomaaaaaaa!!!!!!!!!")
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df['col2'].count(), i]], ignore_index=True)
    print(v_set[counter])
counter += 1
"""

# 0. FI curve
v_set[counter] = df.ix[:df["Section[1] (mV)"].count(), 1]
for i in range(2, 21):
    text = "Section[" + str(i) + "] (mV)"
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df[text].count(), i]], ignore_index=True)
    print(v_set[counter])
    print(df.ix[:df[text].count(), i])
counter += 1

# 1. state
v_set[counter] = df.ix[:df["Section[22] (mV)"].count(), 22]
for i in range(23, 42):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 2. V-clamp
v_set[counter] = df.ix[:df["Section[43] (mV)"].count(), 43]
for i in range(44, 93):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 3. state
v_set[counter] = df.ix[:df["Section[94] (mV)"].count(), 94]
for i in range(95, 113):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 4. FI curve
v_set[counter] = df.ix[:df["Section[114] (mV)"].count(), 114]
for i in range(115, 133):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 5. FI curve
v_set[counter] = df.ix[:df["Section[134] (mV)"].count(), 134]
for i in range(135, 153):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 6. FI curve
v_set[counter] = df.ix[:df["Section[154] (mV)"].count(), 154]
for i in range(155, 173):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 7. FI curve
v_set[counter] = df.ix[:df["Section[174] (mV)"].count(), 174]
for i in range(175, 193):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 8. state
v_set[counter] = df.ix[:df["Section[194] (mV)"].count(), 194]
for i in range(195, 197):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 9. FI curve
v_set[counter] = df.ix[:df["Section[198] (mV)"].count(), 198]
for i in range(199, 217):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 10. FI curve
v_set[counter] = df.ix[:df["Section[218] (mV)"].count(), 218]
for i in range(219, 237):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 11. FI curve
v_set[counter] = df.ix[:df["Section[238] (mV)"].count(), 238]
for i in range(239, 257):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 12. FI curve
v_set[counter] = df.ix[:df["Section[258] (mV)"].count(), 258]
for i in range(259, 277):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 13. state(2 cells)
v_set[counter] = df.ix[:df["Section[278] (mV)"].count(), 278]
for i in range(279, 297):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 14. state(single cell)
v_set[counter] = df.ix[:df["Section[298] (mV)"].count(), 298]
for i in range(299, 307):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 15. V-clamp
v_set[counter] = df.ix[:df["Section[308] (mV)"].count(), 308]
for i in range(309, 316):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 16. FI curve
v_set[counter] = df.ix[:df["Section[317] (mV)"].count(), 317]
for i in range(318, 336):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 17. state
v_set[counter] = df.ix[:df["Section[337] (mV)"].count(), 337]
for i in range(338, 351):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 18. FI curve
v_set[counter] = df.ix[:df["Section[352] (mV)"].count(), 352]
for i in range(353, 371):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 19. state
v_set[counter] = df.ix[:df["Section[372] (mV)"].count(), 372]
for i in range(373, 376):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 20. V-clamp
v_set[counter] = df.ix[:df["Section[377] (mV)"].count(), 377]
for i in range(378, 390):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 21. FI curve
v_set[counter] = df.ix[:df["Section[391] (mV)"].count(), 391]
for i in range(392, 410):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 22. FI curve
v_set[counter] = df.ix[:df["Section[411] (mV)"].count(), 411]
for i in range(412, 430):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 23. FI curve
v_set[counter] = df.ix[:df["Section[431] (mV)"].count(), 431]
for i in range(432, 450):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 24. FI curve
v_set[counter] = df.ix[:df["Section[451] (mV)"].count(), 451]
for i in range(452, 470):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 25. FI curve
v_set[counter] = df.ix[:df["Section[471] (mV)"].count(), 471]
for i in range(472, 490):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 26. FI curve
v_set[counter] = df.ix[:df["Section[491] (mV)"].count(), 491]
for i in range(492, 510):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1
print("kanopero")
# 27. FI curve
v_set[counter] = df.ix[:df["Section[511] (mV)"].count(), 511]
for i in range(512, 530):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 28. FI curve
v_set[counter] = df.ix[:df["Section[531] (mV)"].count(), 531]
for i in range(532, 553):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)
counter += 1

# 29. V-clamp
v_set[counter] = df.ix[:df["Section[554] (mV)"].count(), 554]
for i in range(555, 560):
    v_set[counter] = pd.concat([v_set[counter], df.ix[:df["Section[" + str(i) + "] (mV)"].count(), i]], ignore_index=True)

for i in range(0, num_sim):
    t = np.arange(0, len(v_set[i]), 1)
    df = pd.DataFrame({'index': t, 'V[mV]': v_set[i]})
    df.to_csv('C:/Users/Hattori/Documents/Andor Solis/' + str(i) + '.csv')
    print(str(i) + "steps")

elapsed_time = time.time() - starttime
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
