# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:26:56 2017

@author: Nakanishi
"""

import matplotlib.pyplot as plt
import pandas as pd


def main():
    hh = pd.read_csv("pnoise0.1-ave5-gh0.04-syng-0.1-numsyn1-10.csv", delimiter=',')
    vin = hh['vin'].as_matrix()
    vde = hh['vde'].as_matrix()
    isyn = hh['isyn'].as_matrix()
    t = hh['t'].as_matrix()
    plt.plot(t, vin)
    
    plt.plot(t, vde)
    '''
    plt.plot(t, isyn)
    '''
if __name__=='__main__':
    main()