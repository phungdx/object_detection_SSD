import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Function
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from math import sqrt
import time
from tqdm import tqdm

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)
