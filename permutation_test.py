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

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

def permutation_test(c_measure, cond1, cond2, freq, length):
    
    if c_measure == 'ccorr':
        
        c1 = []
        c2 = []
        
        pairs = ['pair003_','pair004_','pair005_','pair007_','pair009_','pair0010_']
        
        path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\ccorr"
        os.chdir(path)
        
        for i in pairs:
            
            for root, dirs, files in os.walk(path):
                
                for f in files:
                    
                    if f.startswith(i + freq + '_' + cond1 + '_' + length + '.npy'):
                        
                        c1.append(np.load(f))
                        
        for i in pairs:
            
            for root, dirs, files in os.walk(path):
                
                for f in files:
                    
                    if f.startswith(i + freq + '_' + cond2 + '_' + length + '.npy'):
                        
                        c2.append(np.load(f))
        
        data = [np.array(c1), np.array(c2)]
        
        statscondCluster = stats.statscondCluster(data=data,
                                          freqs_mean=np.arange(4, 25),
                                          ch_con_freq=None,
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)
        
    return statscondCluster
