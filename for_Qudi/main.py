# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:56:09 2017

@author: Hattori
"""

import serial
import pandas as pd
import csv

ser = serial.Serial()
ser.baudrate = 9600
ser.port = 3
ser.open()

f = open('test.csv', 'w')
writer = csv.writer(f)

while(True):
    line = str(ser.read(8))
    print(line)
    writer.writerow(line)

f.close()
ser.close()
