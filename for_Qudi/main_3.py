# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:35:49 2017

@author: Hattori
"""
import time
import serial
import csv


def main():

    # byte type
    text = b'kanoperojupippikanoperojupippikanoperojupippikanoperojupippi_non'

    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 2
    ser.open()
    """
    ser2 = serial.Serial()
    ser2.baudrate = 115200
    ser2.port = 1
    ser2.open()
    """
    f = open('test3.csv', 'w')
    writer = csv.writer(f)
    csvlist = [[]]

    while(True):
        ser.write(text)
        ser.flush()

        # byte to str
        line = (ser.read(64)).decode('utf-8')
        print(line)
        csvlist[0] = line
        writer.writerow(csvlist)
        #time.sleep(0.01)

    f.close()
    # ser2.close()
    ser.close()

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

