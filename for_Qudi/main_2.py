# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:35:49 2017

@author: Hattori
"""
import datetime
import time
import serial
import csv
import random

def main():

    # byte type
    #text = b'kanoperojupippikanoperojupippikanoperojupippikanoperojupippi_non'a

    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM4'
    ser.open()
    """
    ser2 = serial.Serial()
    ser2.baudrate = 115200
    ser2.port = 1
    ser2.open()
    """
    f = open('test3.txt', 'w')
    writer = csv.writer(f)
    csvlist = [[]]

    source_str = 'abcdefghijklmnopqrstuvwxyz0123456789'

    while(True):
    #for i in range(0, 1000):
        
        # text format
        # text = b'kashikoma'
        text = "".join([random.choice(source_str) for x in range(511)])
        text = text.encode('utf-8')
        # text = str(datetime.datetime.today()).encode('utf-8')
        
        csvlist[0] = 'send: ' + str(text)
        writer.writerow(csvlist)
        
        ser.flushOutput()
        ser.write(text)
        ser.flush()

        # byte to str
        # line = (ser.read(64)).decode('utf-8')
        line = ser.read(512)
        print(line)
        csvlist[0] = 'receive: ' + str(line)
        writer.writerow(csvlist)
        ser.flushInput()
        #time.sleep(0.01)

    f.close()
    # ser2.close()
    ser.close()

if __name__ == '__main__':
    main()
