# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:24:54 2017

@author: aromagedon
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import time
from neuron5 import Neuron5 as Neuron

class Main():
    def plot(self,):

        self.neuron = Neuron()        
        self.fig, self.ax = plt.subplots(nrows = self.neuron.numneu, figsize=(10, 10))
        
        #グラフ設定
        for i in range(0, self.neuron.numneu):
            if self.neuron.numneu == 1:
                self.ax.set_xlim((0, self.neuron.simtime))
                self.ax.set_ylim(-80, 40)
                self.ax.grid(which='major',color='thistle',linestyle='-')
                self.ax.spines["top"].set_color("indigo")
                self.ax.spines["bottom"].set_color("indigo")
                self.ax.spines["left"].set_color("indigo")
                self.ax.spines["right"].set_color("indigo")

            else:            
                self.ax[i].set_xlim((0, self.neuron.simtime))
                self.ax[i].set_ylim(-80, 40)
                self.ax[i].grid(which='major',color='thistle',linestyle='-')
                self.ax[i].spines["top"].set_color("indigo")
                self.ax[i].spines["bottom"].set_color("indigo")
                self.ax[i].spines["left"].set_color("indigo")
                self.ax[i].spines["right"].set_color("indigo")
        
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, bottom=0.03)

        self.fig.text(0.5, 0.01, 'time [t]', fontsize=16, ha='center', va='center')        
        self.fig.text(0.01, 0.5, 'membrane potential [mV]', fontsize=16, ha='center', va='center', rotation='vertical')        
        self.fig.patch.set_facecolor('whitesmoke')   
        
        #初期化
        self.lines = []
        for i in range(0, self.neuron.numneu):
            self.lines.append([])
            if self.neuron.numneu == 1:
                self.lines, = self.ax.plot(self.neuron.tmhist, self.neuron.vin[i], color="indigo")
            else:
                self.lines[i], = self.ax[i].plot(self.neuron.tmhist, self.neuron.vin[i], color="indigo")
        
        #信号伝達 
        self.start_time = time.time()
        for i in range(0, np.size(self.neuron.tmhist)-1):      
            self.neuron.propagation()
        self.elapsed_time = time.time() - self.start_time
        print("elapsed_time:{0}".format(self.elapsed_time) + "[sec]")       
       
        #プロット
        for i in range(0, self.neuron.numneu):        
            base = np.ones(np.size(self.neuron.tmhist)) * (-100)            
            if self.neuron.numneu == 1:
                self.lines.set_data(self.neuron.tmhist, self.neuron.vin[i])
                self.ax.fill_between(self.neuron.tmhist, self.neuron.vin[i], base, facecolor="thistle", alpha=0.2)           
            else:
                self.lines[i].set_data(self.neuron.tmhist, self.neuron.vin[i])              
                self.ax[i].fill_between(self.neuron.tmhist, self.neuron.vin[i], base, facecolor="thistle", alpha=0.2)                                 


def main():
    main = Main()
    main.plot()

    
if __name__ == '__main__':
    main()