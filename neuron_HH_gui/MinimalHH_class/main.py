# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:23:10 2018

@author: ishida
"""
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from neuron_class import Neuron_HH

save_path = "C:/Users/ishida/Documents/simulation"

def main():
    neuron = Neuron_HH(syncp = 1)
    
    for i in range(0, neuron.allsteps - 1):
        neuron.propagation()
        

    d = datetime.datetime.today()
    filename = (str(d.year) + "_" + str(d.month) + "_" +
                str(d.day) + "_" + str(d.hour) + "_" + str(d.minute) +
                "_" + str(d.second) + "_" + "HHclass.csv")
        
    df = pd.DataFrame({"T[ms]" : neuron.Tsteps, 
                       "V[mV]" : neuron.V[0],
                       "INa[uA/cm^2]" : neuron.INa[0],
                       "IK[uA/cm^2]" : neuron.IK[0],
                       "Im[uA/cm^2]" : neuron.Im[0],
                       "ItCa[uA/cm^2]" : neuron.ItCa[0],
                       "Ileak[uA/cm^2]" : neuron.Ileak[0],
                       "Isyn[uA/cm^2]" : neuron.Isyn[0],
                       "Inoise[uA/cm^2]" : neuron.Inoise[0]})
    df.to_csv(save_path + '/' + filename)
    
    
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.plot(neuron.Tsteps, neuron.V[0])
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("voltage [mV]")

    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax2.plot(neuron.Tsteps, neuron.Ileak[0], label = "Ileak")
    ax2.plot(neuron.Tsteps, neuron.INa[0], label = "INa")
    ax2.plot(neuron.Tsteps, neuron.IK[0], label = "IK")
    ax2.plot(neuron.Tsteps, neuron.Im[0], label = "Im")
    ax2.plot(neuron.Tsteps, neuron.ItCa[0], label = "ItCa")
    ax2.plot(neuron.Tsteps, neuron.Isyn[0], label = "Isyn")
    ax2.plot(neuron.Tsteps, neuron.Inoise[0], label = "Inoise")
    ax2.legend()
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("current [uA/cm^2]")
    plt.show()    

if __name__ ==  '__main__':
    main()