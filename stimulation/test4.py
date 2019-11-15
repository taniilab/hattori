import serial
import numpy as np
from time import sleep

port_number = "COM11"

ser = serial.Serial(port_number, 115200, timeout=1)
text = "COM" + str(11) + " Opened!!"
print(text)

"""accurately
for i in range(10):
    command = ("WMW0" + str(i)+ "\n")
    ser.write(command.encode())
    print("WMW0" + str(i))
    sleep(1.0)

for i in range(10):
    command = ("WMW1" + str(i)+ "\n")
    ser.write(command.encode())
    print("WMW1" + str(i))
    sleep(1.0)
"""
command = ("WMW012" + "\n")
ser.write(command.encode())
sleep(3)
"""
command = ("WMW013" + "\n")
ser.write(command.encode())
sleep(3)

command = ("WMW014" + "\n")
ser.write(command.encode())
sleep(3)

command = ("WMW015" + "\n")
ser.write(command.encode())
sleep(3)

command = ("WMW012" + "\n")
ser.write(command.encode())
sleep(3)
"""

print("kanopero")
