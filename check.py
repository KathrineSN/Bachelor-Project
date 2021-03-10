# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:36:03 2021

@author: kathr
"""

import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from hypyp import prep 
from hypyp import analyses
from hypyp import stats
from hypyp import viz
from collections import Counter

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

epochs_a = mne.read_epochs('epochs_a_short.fif', preload = True)
epochs_b = mne.read_epochs('epochs_b_short.fif', preload = True)