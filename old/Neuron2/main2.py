                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     # -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:45:33 2016

@author: aromagedon
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from neuron2 import Neuron2


deltatime = 0.05
def main():
    global deltatime

    x = [1.0, 0.3, 0.3, 0.3, 0.3]
    y = [1.0, 1.0, 0.3, 0.3, 0.3]
    z = [2.0, 2.0, 0.0, 0.3, 0.3]
    p = [3.0, 3.0, 0.3, 0.0, 0.3]
    q = [1.5, 1.5, 0.3, 0.3, 0.0]
    
    #コンストラクタ引数(self, num_neurons, vth, vrm, tau, def_w, phigh, deltatime)
    neuron1 = Neuron2(5, -40, -70, 3, 0.5, 40, deltatime)
    neuron1.set_weight(np.array((x, y, z, p, q), dtype=float))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, figsize=(14,16))
    t = np.arange(0, 5, deltatime)
    v1out = [-70] * np.size(t)
    v2out = [-70] * np.size(t)
    v3out = [-70] * np.size(t)
    v4out = [-70] * np.size(t)
    v5out = [-70] * np.size(t)
    
    print(neuron1.numneu)
    print(neuron1.vth)
    print(neuron1.vrm)
    print(neuron1.tau)
    print(neuron1.phigh)
    print(neuron1.f_flag)
    print(neuron1.input)
    print(neuron1.output)
    print(neuron1.vin)
    print(neuron1.n)
    print(neuron1.w)
    print(neuron1.sigma)
    print(neuron1.temp)

    #　配列末尾
    last1 = len(v1out)-1  
    last2 = len(v2out)-1    
    last3 = len(v3out)-1
    last4 = len(v4out)-1    
    last5 = len(v5out)-1
    
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    lines1, = ax1.plot(t, v1out)
    lines2, = ax2.plot(t, v2out)
    lines3, = ax3.plot(t, v3out)
    lines4, = ax4.plot(t, v4out)
    lines5, = ax5.plot(t, v5out)
    
    #グラフ設定
    ax1.set_title("Neuron1") 
    ax2.set_title("Neuron2") 
    ax3.set_title("Neuron3") 
    ax4.set_title("Neuron4") 
    ax5.set_title("Neuron5")
    ax1.set_ylabel("membrane potential [mV]")
    ax1.set_xlabel("time [t]")
    ax2.set_ylabel("membrane potential [mV]")
    ax2.set_xlabel("time [t]")
    ax3.set_ylabel("membrane potential [mV]")
    ax3.set_xlabel("time [t]")
    ax4.set_ylabel("membrane potential [mV]")
    ax4.set_xlabel("time [t]")
    ax5.set_ylabel("membrane potential [mV]")
    ax5.set_xlabel("time [t]")
    fig.tight_layout()    
    
    f = open('neuron.csv', 'ab')
  
    while True:
        #グラフ用        
        t += deltatime
        v1out.append(0)
        v1out.pop(0)
        v2out.append(0)
        v2out.pop(0)        
        v3out.append(0)
        v3out.pop(0)
        v4out.append(0)
        v4out.pop(0)        
        v5out.append(0)
        v5out.pop(0)
        
        #注入
        neuron1.input[0][0] = 20
        neuron1.input[1][1] = 20
       
        vin, output = neuron1.propagation("sigmoid")
        tex = str(vin[0]) + '\n'
        f.write(tex.encode('utf-8'))

        #グラフ用
        v1out[last1] = vin[0] 
        v2out[last2] = output[0] 
        v3out[last3] = vin[2] 
        v4out[last4] = vin[3] 
        v5out[last5] = vin[4] 
        
        print(vin)
        print(output)
        print()
        
        #グラフ更新
        lines1.set_data(t, v1out)
        ax1.set_xlim((t.min(), t.max()))
        ax1.set_ylim(-90, 50)
        lines2.set_data(t, v2out)
        ax2.set_xlim((t.min(), t.max()))
        ax2.set_ylim(-90, 50)
        lines3.set_data(t, v3out)
        ax3.set_xlim((t.min(), t.max()))
        ax3.set_ylim(-90, 50)
        lines4.set_data(t, v4out)
        ax4.set_xlim((t.min(), t.max()))
        ax4.set_ylim(-90, 50)
        lines5.set_data(t, v5out)
        ax5.set_xlim((t.min(), t.max()))
        ax5.set_ylim(-90, 50)
        
        plt.pause(.01)
        
if __name__ == "__main__":
    main()
