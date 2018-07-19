# coding: UTF-8
from multiprocessing import Pool
import os
import numpy as np
import time
import datetime
import itertools
i = 3
j = 3
k = 3

for i, j, k in itertools.product(range(i),
                                 range(j),
                                 range(k)):

    result = (i+1)*100 + (j+1)*10 + (k+1)
    print(result)
