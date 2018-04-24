# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:13:33 2018

@author: Hattori
"""
from car import CAR

def main():
    c = CAR(1416)
    print(c.light)

    c.light_ctrl(True)
    print(c.light)


if __name__ == '__main__':
    main()

