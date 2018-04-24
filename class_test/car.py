# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:44 2018

@author: Hattori
"""

class CAR:
    num_hundle = 1

    #コンストラクタ
    def __init__(self, n = 1602):
        self.number = n
        self.light = "OFF"

    def light_ctrl(self, flag=False):
        self.flag = flag
        
        if self.flag == True:
            self.light = "ON"
        else:
            self.light = "OFF"
