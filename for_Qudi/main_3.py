# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:35:49 2017

@author: Hattori
"""
import time
import serial
import numpy as np

def main():

    # byte type
    # text = b'kanoperojupippikanoperojupippikanoperojupippikanoperojupippi_non'
    
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = 2
    ser.open()

    cycle = 1000
    volt_max = 1.3

    while(True):

        for i in range(0, cycle):
            t = 2 * np.pi * i / cycle
            x = np.sin(t) ** 2
            y = np.sin(t + (np.pi/2)) ** 2
            z = np.sin(t + np.pi) ** 2
            print(x)
            print(y)
            print(z)
            print()
            data_x = (x / volt_max) * 65536
            data_y = (y / volt_max) * 65536
            data_z = (z / volt_max) * 65536
            print(data_x)
            print(data_y)
            print(data_z)
            print(int(data_x))
            print(int(data_y))
            print(int(data_z))
            print("kanoepro")
            
            data_x = int(data_x)
            data_y = int(data_y)
            data_z = int(data_z)
        

            data = 256*(data_x +
                    data_y*65536 +
                    data_z*65536*65536)
            
            data_bytes = data.to_bytes(8, 'big')

            print(hex(data_x))
            print(hex(data_y))
            print(hex(data_z))            
            print(data)
            print(hex(data))
            print(data_bytes)
            
            print("kashikoma")
            print()
            ser.write(data_bytes)
            ser.flush()

            
        """
        ser.write(text)
        ser.flush()

        # byte to str
        line = (ser.read(64)).decode('utf-8')
        print(line)
        csvlist[0] = line
        writer.writerow(csvlist)
        # time.sleep(0.01)
        """        
    ser.close()


if __name__ == '__main__':
    main()
