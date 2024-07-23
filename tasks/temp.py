import os.path
import shutil
from typing import Any, Dict

import sys
sys.path.append('/home/liupei/code/WcDT/')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch


##test 
f = open('/home/liupei/code/WcDT/val.txt', 'r')
res = []
for line in f.readlines():
    # print(line)
    line = list(map(float, line.split(',')[:-1]))
    res.extend(line)

print(np.mean(res))
