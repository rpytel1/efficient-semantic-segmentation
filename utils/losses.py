"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def get_class_weights():
    weights = []
    with open('utils/weights.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        weights.append(int(line))

    return torch.FloatTensor(weights).cuda()

weights = get_class_weights()







