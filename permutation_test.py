# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:55:57 2021

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
from collections import OrderedDict
from itertools import groupby

#epochs = mne.read_epochs('epochs_a_short_007.fif', preload = True)
#epochs.set_montage('biosemi64')
#adj = mne.channels.find_ch_adjacency(info = epochs.info, ch_type = 'eeg')

c1 = np.load('pair003_alpha_Control_short.npy')
c2 = np.load('pair004_alpha_Control_short.npy')
c3 = np.load('pair005_alpha_Control_short.npy')
c4 = np.load('pair007_alpha_Control_short.npy')
c5 = np.load('pair009_alpha_Control_short.npy')
c6 = np.load('pair0010_alpha_Control_short.npy')

co1 = np.load('pair003_alpha_Coupled_short.npy')
co2 = np.load('pair004_alpha_Coupled_short.npy')
co3 = np.load('pair005_alpha_Coupled_short.npy')
co4 = np.load('pair007_alpha_Coupled_short.npy')
co5 = np.load('pair009_alpha_Coupled_short.npy')
co6 = np.load('pair0010_alpha_Coupled_short.npy')




data = [np.array([co1, co2, co3, co4, co5, co6]), np.array([c1, c2, c3, c4, c5, c6])]

permutation_test = mne.stats.permutation_cluster_test(data, threshold = 0.2)
#Look at the 0th index to get f-statistics

'''
statscondCluster = stats.statscondCluster(data=data,
                                          freqs_mean=np.arange(4, 25),
                                          ch_con_freq=None,
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=1.0)
'''