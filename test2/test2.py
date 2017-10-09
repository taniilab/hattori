# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:39:53 2017

@author: Hattori
"""

palm = {"Syncp": 3, "Iext": 4}
palm = str(palm)
print(palm)

palm = palm.replace(':', '_')
palm = palm.replace('{', '_')
palm = palm.replace('}', '_')
palm = palm.replace('\'', '')
palm = palm.replace(',', '_')
print(palm)