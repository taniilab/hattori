# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:05:35 2018

@author: Hattori
"""

import serial as s
import sys


class Main():
    def __init__(self):
        self.ser = s.Serial("COM7", 9600)
        print(self.ser.name)


if __name__ == '__main__':
    main = Main()
    
    while(1):
        main.ser.write(b'1')
        data_b = main.ser.read(2)
        data_i = int.from_bytes(data_b, 'little')
        voltage = 3.3 * (data_i / 2**10)
        text = str(round(voltage, 2)) + " V"
        print(text)
    sys.exit()
