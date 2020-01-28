"""
date: 20200106
created by: ishida
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import itertools
import os
from multiprocessing import Pool
from stim_neuron2 import Neuron
from stim_neuron2 import PotentialStimulation

starttime = time.time()
save_path = '//192.168.13.10/Public/ishida/simulation'


class Main:
    def __init__(self):
