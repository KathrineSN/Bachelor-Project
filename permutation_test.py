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
import scipy

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

adj = scipy.sparse.load_npz(path + '\\Adjacency\\adjacency.npz')

#Using t-test instead of default F-test
def ttest_no_p(*args):
    tvals, _ = scipy.stats.ttest_ind(*args)
    return tvals
      

def permutation_test(c_measure, cond1, cond2, freq, length):
    
    if c_measure == 'ccorr':
        
        c1 = []
        c2 = []
        
        pairs = ['pair003_','pair004_','pair005_','pair007_','pair009_','pair0010_', 'pair0011_', 'pair0012_', 'pair0014_', 'pair0016_', 'pair0017_', 'pair0018_']
        
        if length == 'long':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\ccorr"
            os.chdir(path)
            print('blob')
            for i in pairs:
                files = os.listdir(path)
                    
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                        
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond2 + '_' + length + '.npy'):
                        print('hej')
                        c = np.triu(np.load(f))
                        print(str(f))
                        c = c[c != 0]
                        c2.append(c)
                        print(len(c1))
        print('test')
        print(len(c1))            
        if length == 'short':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\ccorr"
            os.chdir(path)
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                            
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond2 + '_' + length + '.npy'):
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c2.append(c)
                        
        if length == '3sec':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\ccorr"
            os.chdir(path)
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + '_' + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                            
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + '_' + freq + '_' + cond2 + '_' + length + '.npy'):
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        print(str(f))
                        c2.append(c)
        
        data = [np.array(c1), np.array(c2)]
        
    if c_measure == 'coh':
        
        c1 = []
        c2 = []
        
        pairs = ['pair003_','pair004_','pair005_','pair007_','pair009_','pair0010_', 'pair0011_', 'pair0012_', 'pair0014_', 'pair0016_', 'pair0017_', 'pair0018_']
        
        if length == 'long':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\coh"
            os.chdir(path)
            print('blob')
            for i in pairs:
                files = os.listdir(path)
                    
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                        
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond2 + '_' + length + '.npy'):
                        print('hej')
                        c = np.triu(np.load(f))
                        print(str(f))
                        c = c[c != 0]
                        c2.append(c)
                        
        if length == 'short':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\coh"
            os.chdir(path)
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                            
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + freq + '_' + cond2 + '_' + length + '.npy'):
                        c = np.triu(np.load(f))
                        print(str(f))
                        c = c[c != 0]
                        c2.append(c)
        
        if length == '3sec':
            path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\coh"
            os.chdir(path)
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + '_' + freq + '_' + cond1 + '_' + length + '.npy'):
                        # Avoiding unnessecary tests by only using upper triangle values in a list
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        c1.append(c)
                            
            for i in pairs:
                files = os.listdir(path)
                for f in files:
                    
                    if f.startswith(c_measure + '_' + i + '_' + freq + '_' + cond2 + '_' + length + '.npy'):
                        c = np.triu(np.load(f))
                        c = c[c != 0]
                        print(str(f))
                        c2.append(c)
        
        data = [np.array(c1), np.array(c2)]
        print(len(data[0]))
        '''
        statscondCluster = stats.statscondCluster(data=data,
                                          freqs_mean=np.arange(4, 25),
                                          ch_con_freq=None,
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)
        '''
    

    #statscondCluster = mne.stats.permutation_cluster_test(X = data, threshold = 2.2281, n_permutations = 5000, tail = 0, stat_fun = ttest_no_p, adjacency = adj)
    statscondCluster = mne.stats.permutation_cluster_test(X = data, threshold =  2.074, n_permutations = 5000, tail = 0, stat_fun = ttest_no_p, adjacency = adj)

    print('test..............')
    print(len(data))
    print(len(data[0]))    

    # Only return of there are significant clusters
    sig_pv = []    
    for i in range(len(statscondCluster[2])):
        if statscondCluster[2][i] < 0.05:
            sig_pv.append(statscondCluster[2][i])
    if sig_pv != []:
        return statscondCluster, data
    else:
        return

