# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:04:41 2017

@author: Hattori
"""

import os
import glob
import shutil

os.chdir('C:/Users/Hattori/Documents/HR_outputs/results')

for i in range(30):
    os.mkdir('pippi' + str(i))

allfiles = glob.glob('C:/Users/Hattori/Documents/HR_outputs/results/*.csv')
"""
path = 'C:/Users/Hattori/Documents/HR_outputs/results/'
for i in allfiles.count/30:
    shutil.move(path+'')
 """
 