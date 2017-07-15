import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2 cells burst 13.csv
#single cell burst 14.csv
path = 'C:/Users/Hattori/Documents/Andor Solis/14.csv'

df = pd.read_csv(path, delimiter=',')
plt.figure(figsize=(10, 7))
plt.plot(df['index'], df['V[mV]'])
plt.show()
