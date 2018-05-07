# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:39:53 2017

@author: Hattori
"""

from PIL import Image

palm = {"Syncp": 3, "Iext": 4}
palm = str(palm)
print(palm)

palm = palm.replace(':', '_')
palm = palm.replace('{', '_')
palm = palm.replace('}', '_')
palm = palm.replace('\'', '')
palm = palm.replace(',', '_')
print(palm)

img = Image.open("./kanon.jpg")
imgW = img.size[0]
imgH = img.size[1]

for y in range(0, imgH):
    for x in range(0, imgW):
        offset = y*imgW + x
        xy = (x, y)
        rgb = img.getpixel(xy)
        rgbR = rgb[0]
        rgbG = rgb[1]
        rgbB = rgb[2]
