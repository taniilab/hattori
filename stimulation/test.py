import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyautogui as pag

fig = plt.figure()

def plot(data):
    plt.cla()                      # 現在描写されているグラフを消去
    rand = np.random.randn(100)    # 100個の乱数を生成
    im = plt.plot(rand)            # グラフを生成



x, y = pag.position()
print(pag.pixel(x, y))
ani = animation.FuncAnimation(fig, plot, interval=10)
plt.show()
