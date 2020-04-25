#メモリ節約テスト

import numpy as np
import pandas as pd

save_path = "Z:/simulation/test"

time = 100000
lump = 10
cycle = int(time/lump)
x = np.zeros((5, lump))
df = pd.DataFrame(columns=['number'])
print(df)
df.to_csv(save_path + "/test.csv", mode='a')

for j in range(cycle):
    for i in range(lump-1):
        x[:, i+1] = x[:, i]+1
    print(x)

    df = pd.DataFrame()
    df['number'] = x[0, :]
    df.to_csv(save_path + "/test.csv", mode='a', header=None)

    x = np.fliplr(x)

