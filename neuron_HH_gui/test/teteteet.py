import numpy as np
import pandas as pd


x = np.zeros((4, 3))
list_c = []
list_c.append("a")
list_c.append("b")
list_c.append("c")

df = pd.DataFrame(x, columns=list_c, index=["v","w","x","y"])
df.at["x", "b"] = 5
df.to_csv("Z:/Box Sync/Personal/tmp_data/aaaaa.csv")
print(df)