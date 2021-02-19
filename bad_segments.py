# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:29:49 2021

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

path="C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt"
os.chdir(path)

epochs_a = mne.read_epochs('epochs_a_preprocessed-epo.fif', preload = True)
epochs_b = mne.read_epochs('epochs_b_preprocessed-epo.fif', preload = True)


