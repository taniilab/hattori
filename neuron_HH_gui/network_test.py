@@ -1,15 +0,0 @@
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.arange(0, 10, 0.1)
y = np.sin(x)
path = '//192.168.13.10/Public/hattori/'

fig = plt.figure(figsize=(20, 10))
plt.plot(x, y)
plt.show()

df = pd.DataFrame({'x':x, 'y':y})
df.to_csv(path + 'test.csv')