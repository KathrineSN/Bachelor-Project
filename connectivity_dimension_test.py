# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:46:50 2021

@author: kathr
"""

import numpy as np
import os
import mne
import scipy

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Connectivity matrices\\ccorr\\"
os.chdir(path)


con = np.load("ccorr_pair003_alpha_Coupled_short.npy")

#avoiding repeats

upper_con = np.triu(con)
upper_con = upper_con[upper_con != 0]

n = np.count_nonzero(upper_con)

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

adj = scipy.sparse.load_npz(path + '\\Adjacency\\adjacency.npz')






