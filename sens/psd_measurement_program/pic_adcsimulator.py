# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:05:35 2018

@author: Hattori
"""

import serial as s
import sys
import numpy as np
from time import sleep

counter2 = 0

class Main():
    def __init__(self):
        self.ser = s.Serial("COM10", 9600)
        print(self.ser.name)
        self.counter = 0
        self.t = np.arange(0, 6.28, 0.05)
        self.t_len = len(self.t)

if __name__ == '__main__':
    main = Main()

    while (1):
        vx1 = int((np.sin(main.t[main.counter])+1)*2048)
        vx2 = int((np.sin(main.t[main.counter]+np.pi/2)+1)*2048)
        vy1 = int((np.sin(main.t[main.counter]+np.pi)+1)*2048)
        vy2 = int((np.sin(main.t[main.counter]+3*np.pi/2)+1)*2048)

        data =  b"\x74" + b"\x6f" + b"\x70" + vx1.to_bytes(2, "big") + vx2.to_bytes(2, "big") + vy1.to_bytes(2, "big") + vy2.to_bytes(2, "big")
        main.ser.write(data)
        print(data)
        print(counter2)

        main.counter += 1
        if main.counter >= main.t_len:
            main.counter = 0
        #sleep(0.02)
        counter2 += 1

    sys.exit()
