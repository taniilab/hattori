import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/Users/Hattori/Documents/Andor Solis/testttttt.csv'

df = pd.read_csv(path, delimiter=',')
plt.figure()
plt.plot(df['steps'], df['V[mV]'])
plt.show()
