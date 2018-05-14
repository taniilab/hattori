# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:34:25 2018

@author: Hattori
"""

import numpy as np



x = np.arange(0, 10, 1)

print(x)

xx = x[3:5]
 
xx[:] = [0, 0]

print(x)
