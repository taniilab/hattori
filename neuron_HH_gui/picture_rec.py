# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:54:07 2017

@author: Hattori
"""
import matplotlib.pyplot as plt
import glob
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import datetime

class Csv():
    def __init__(self, path):

        self.nowdir = path
        self.csvs = path + '/' + '*.csv'
        self.files = {}
        self.counter = 0
        self.files = glob.glob(self.csvs)

    def run(self):
        for file_ in self.files:
            df = pd.read_csv(file_, index_col=0)
            df.plot(x='time[ t]', y='	voltage [mV]', lw=0.5)
            print(str(self.counter) + '個目のファイルを処理します')
            self.counter += 1
        plt.show()

path = "C:/Users/Hattori/Desktop/test"
c = Csv(path)
c.run()
