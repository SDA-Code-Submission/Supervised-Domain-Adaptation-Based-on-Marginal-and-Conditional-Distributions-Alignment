import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
import torchvision.datasets as torchdatasets
from torch.utils.data import ConcatDataset

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import sklearn.datasets as dt
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import pdb
from tqdm import tqdm
import random
from PIL import Image
import xml.etree.ElementTree as ET
from glob import glob
from typing import Optional, Callable, Tuple, Any, List
import timm
import wandb

import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)