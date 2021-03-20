# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:51:56 2021

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

def load_avg_matrix(freq_band, cond, length, plot = 1, sep = 0, save = 0):
    
    matrices = []
    
    pairs = ['pair003_','pair004_','pair005_','pair007_','pair009_','pair0010_']
    
    for i in pairs:
        
        for root, dirs, files in os.walk(path):
            
            for f in files:
                
                if f.startswith(i + freq_band + '_' + cond + '_' + length + '.npy'):
                    
                    matrices.append(np.load(f))
                    
    #avg_matrix = np.mean(matrices, axis = )
    mat_sum = np.zeros_like(matrices[0])
    for mat in matrices:
        mat_sum += mat
    avg_matrix = mat_sum/len(matrices)
    
    if plot:
        fig = plt.figure()
        plt.title(cond + ' ' + freq_band + ' ' + length)
        plt.imshow(avg_matrix,cmap=plt.cm.hot)
        plt.clim(0,0.8)
        plt.colorbar()
        plt.show()
        if save:
                fig.savefig('avg_matrices/' + cond + '_' + freq_band + '_' + length + '.png')
    
    if sep:
        for i in range(len(matrices)):
            fig = plt.figure()
            plt.title(cond + ' ' + freq_band + ' ' + length)
            plt.imshow(matrices[i], cmap=plt.cm.hot)
            plt.clim(0,0.8)
            plt.colorbar()
            plt.show()
            if save:
                fig.savefig('sep_matrices/' + str(pairs[i]) + '_' + cond + '_' + freq_band + '_' + length + '.png')
         
    return avg_matrix

#%%
#Short epochs
load_avg_matrix('alpha', 'Coupled', 'short', save = 1)
load_avg_matrix('beta', 'Coupled', 'short')
load_avg_matrix('theta', 'Coupled', 'short')
load_avg_matrix('alpha', 'Uncoupled', 'short')
load_avg_matrix('beta', 'Uncoupled', 'short')
load_avg_matrix('theta', 'Uncoupled', 'short')
load_avg_matrix('alpha', 'Control', 'short')
load_avg_matrix('beta', 'Control', 'short')
load_avg_matrix('theta', 'Control', 'short')

#%%
#Long epochs
load_avg_matrix('alpha', 'Coupled', 'long')
load_avg_matrix('beta', 'Coupled', 'long')
load_avg_matrix('theta', 'Coupled', 'long')
load_avg_matrix('alpha', 'Uncoupled', 'long')
load_avg_matrix('beta', 'Uncoupled', 'long')
load_avg_matrix('theta', 'Uncoupled', 'long')
load_avg_matrix('alpha', 'Control', 'long')
load_avg_matrix('beta', 'Control', 'long')
load_avg_matrix('theta', 'Control', 'long')

#%%
#Creating separate plots
load_avg_matrix('alpha','Coupled','short', plot = 0, sep = 1, save = 1)
load_avg_matrix('beta','Coupled','short', plot = 0, sep = 1)
load_avg_matrix('alpha','Uncoupled','short', plot = 0, sep = 1)
load_avg_matrix('theta','Control','short', plot = 0, sep = 1)
load_avg_matrix('alpha','Coupled','long', plot = 0, sep = 1, save = 0)
load_avg_matrix('beta','Coupled','long', plot = 0, sep = 1, save = 0)














