# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:31:37 2016

@author: aromagedon
"""
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from neuron import Neuron


deltatime = 0.10
def main():
    global deltatime

    neuron1 = Neuron(2, 2, 0.5)
    neuron2 = Neuron(2, 2, 0.5)
    neuron3 = Neuron(2, 2, 0.5)
    neuron1.set_weight(np.array([0.4, 0.4]))
    neuron2.set_weight(np.array([0.4, 0.4]))    
    neuron3.set_weight(np.array([0.4, 0.4]))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12,12))
    t = np.arange(0, 200, deltatime)
    v1out = [0] * np.size(t)
    v2out = [0] * np.size(t)
    v3out = [0] * np.size(t)
    
    #　配列末尾
    last1 = len(v1out)-1
    print (len(v1out))
    
    last2 = len(v2out)-1
    print (len(v2out))
    
    last3 = len(v3out)-1
    print (len(v3out))
    
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    lines1, = ax1.plot(t, v1out)
    lines2, = ax2.plot(t, v2out)
    lines3, = ax3.plot(t, v3out)
    
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
        
        #入出力処理
        neuron1.prop_p_linear()
        neuron2.prop_p_linear()
        neuron3.prop_p_linear()
  
      #先頭値更新
        neuron3.vin[1][0] = neuron1.vout[0][0]
        neuron2.vin[0][0] = neuron1.vout[1][0]
        neuron1.vin[1][0] = neuron2.vout[0][0]
        neuron3.vin[0][0] = neuron2.vout[1][0]
        neuron2.vin[1][0] = neuron3.vout[0][0]
        neuron1.vin[0][0] = neuron3.vout[1][0]

        neuron1.set_noise()
        neuron2.set_noise()
        neuron3.set_noise()
       
        """
        print(n1)
        print()
        """        
        """                
        neuron1.n[0][0] = np.random.randn() * 0.25
        neuron2.n[0][0] = np.random.randn() * 0.25
        neuron3.n[0][0] = np.random.randn() * 0.25
        """
        
        tex = str(neuron3.vout[0][0]) + ',' + str(neuron3.vout[1][0]) + '\n'
        f.write(tex.encode('utf-8'))

        #グラフ用
        v1out[last1] = neuron1.vout[0][0] 
        v2out[last2] = neuron2.vout[0][0] 
        v3out[last3] = neuron3.vout[0][0] 
        
  
        #グラフ更新
        lines1.set_data(t, v1out)
        ax1.set_xlim((t.min(), t.max()))
        ax1.set_ylim(-1.,3.0)
        lines2.set_data(t, v2out)
        ax2.set_xlim((t.min(), t.max()))
        ax2.set_ylim(-1.,3.0)
        lines3.set_data(t, v3out)
        ax3.set_xlim((t.min(), t.max()))
        ax3.set_ylim(-1.,3.0)
        
        plt.pause(.01)
        
if __name__ == "__main__":
    main()
