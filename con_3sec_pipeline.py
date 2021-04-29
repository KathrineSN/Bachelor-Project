# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:32:23 2021

@author: kathr
"""
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)
import mne
from collections import OrderedDict
from itertools import groupby
from hypyp import prep 
from con_functions import *


#%% Looping over all pairs for 3 sec

pairs = ['pair003_','pair004_','pair005_','pair007_','pair009_','pair0010_','pair0011_','pair0012_','pair0014_','pair0016_','pair0017_','pair0018_']
no = [3,4,5,7,9,10,11,12,14,16,17,18]

for i in range(len(pairs)):
    epo_a = mne.read_epochs('epochs_a_3sec_' + str(no[i])+ '.fif')
    epo_b = mne.read_epochs('epochs_b_3sec_' + str(no[i])+ '.fif')
    ccorr(epo_a, epo_b, pairs[i], '3sec', drop_list = [])
    epo_a = mne.read_epochs('epochs_a_3sec_' + str(no[i])+ '.fif')
    epo_b = mne.read_epochs('epochs_b_3sec_' + str(no[i])+ '.fif')
    coh(epo_a, epo_b, pairs[i], '3sec', drop_list = [])

#%% Looping over all pairs for short

pairs = ['pair0011','pair0012','pair0014','pair0016','pair0017','pair0018']
no = [11,12,14,16,17,18]

for i in range(len(pairs)):
    epo_a = mne.read_epochs('epochs_a_short_' + str(no[i])+ '.fif')
    epo_b = mne.read_epochs('epochs_b_short_' + str(no[i])+ '.fif')
    ccorr(epo_a, epo_b, pairs[i], 'short', drop_list = [])
    epo_a = mne.read_epochs('epochs_a_short_' + str(no[i])+ '.fif')
    epo_b = mne.read_epochs('epochs_b_short_' + str(no[i])+ '.fif')
    coh(epo_a, epo_b, pairs[i], 'short', drop_list = [])
    

