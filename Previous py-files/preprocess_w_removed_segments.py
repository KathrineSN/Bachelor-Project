# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:04:41 2021

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

# Loading raw
raw = mne.io.read_raw_bdf('Data\\pair003_20200129_1530.bdf', preload=True)

print(raw.info)

# Filtering
f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")

# Loading in annotations

annot = mne.read_annotations('bad_segments_pair003-annot.fif')

f_raw.set_annotations(annot)

f_raw.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
