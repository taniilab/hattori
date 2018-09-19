@@ -1,45 +0,0 @@
import pandas as pd
import glob
import os

path = "G:/rename_test"
csvs = path + '/' + '*.csv'
files = glob.glob(csvs)

print(files)
files.sort()
print(files)

i=0
j=0
k=0
i_limit = round(1.00, 3)
j_limit = round(1.00, 3)
k_limit = round(4.00, 3)

for file in files:
    add_name = "P_AMPA" + str(i) + "_P_NMDA" + str(j) + "_Mg_conc" + str(k)
    os.rename(file, os.path.join(path, add_name + os.path.basename(file)))
    if i < i_limit:
        i = round(i+0.05, 4)
    elif i == i_limit:
        i = 0
        j = round(j+0.05, 4)

    if j == j_limit+0.05:
        j = 0
        k = round(k+0.1, 4)


"""
files = glob.glob(csvs)

#先頭
filename = os.path.basename(files[2])
print(filename)
config = pd.DataFrame(columns=[filename])
config.to_csv(path + '/test/' + filename)

df = pd.read_csv(str(files[2]))
df.to_csv(path + '/test/' + filename, mode='a')
"""