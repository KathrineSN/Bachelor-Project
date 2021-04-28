# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:16:54 2021

@author: kathr
"""
#Importing dependencies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
from mat_functions import *
from con_functions import * 
import matplotlib.pyplot as plt
#from ccorr import ccorr
#from avg_matrices import load_avg_matrix


#%% Preprocessing of long epochs
file = 'pair0018_20200210_1530.bdf'
epo1, epo2 = prepocess_long(file)
#%% Bad channel identification
bads1 = ['PO3']
bads2 = ['P1']
epo1 = bad_removal(epo1, bads1)
epo2 = bad_removal(epo2, bads2)  
#%%
ica1 = ica_part(epo1)
ica2 = ica_part(epo2)
#%%  Bad componenet identification of a
exclude1 = [0,2]
save_name1 = 'epochs_a_long_18.fif'
epo1_c = ica_removal_long(epo1, ica1, exclude1, save_name1)

#%%Bad componenet identification of b
exclude2 = [0,4]
save_name2 = 'epochs_b_long_18.fif'
epo2_c = ica_removal_long(epo2, ica2, exclude2, save_name2)
#%%
before_vs_after_ICA(epo1, epo1, epo1_c, epo2_c)
#%%
ccorr(epo1_c, epo2_c, 'pair003', 'long',drop_list = [])
#%%
plt.close('all')
load_avg_matrix('beta','Coupled','long', plot = 0, sep = 1, save = 0)

#%% Preprocessing of short epochs
file_s = 'pair0018_20200210_1530.bdf'
epo1_s, epo2_s = prepocess_short(file_s)
#%% Bad channel identification
bads1_s = ['PO3']
bads2_s = ['P1']
epo1_s = bad_removal(epo1_s, bads1_s)
epo2_s = bad_removal(epo2_s, bads2_s) 
#%%
plt.close('all')
ica1_s = ica_part(epo1_s)
ica2_s = ica_part(epo2_s)
#%%  Bad componenet identification of a
plt.close('all')
exclude1_s = [0,2]
exclude2_s = [0,4]
save_name1_s = 'epochs_a_short_18.fif'
save_name2_s = 'epochs_b_short_18.fif'
epo1_c_s, epo2_c_s = ica_removal_short(epo1_s, epo2_s, ica1_s, ica2_s, exclude1_s, exclude2_s, save_name1_s, save_name2_s)
#%% Preprocessing of 3 sec. epochs 
file_3s = 'pair0018_20200210_1530.bdf'
epo1_3s, epo2_3s = prepocess_3sec(file_3s)
#%% Bad channel identification
plt.close('all')
bads1_3s = ['PO3']
bads2_3s = ['P1']
epo1_3s = bad_removal(epo1_3s, bads1_3s)
epo2_3s = bad_removal(epo2_3s, bads2_3s) 
#%%
plt.close('all')
ica1_3s = ica_part(epo1_3s)
ica2_3s = ica_part(epo2_3s)
#%%  Bad componenet identification of a
plt.close('all')
exclude1_3s = [0,2]
exclude2_3s = [0,4]
save_name1_3s = 'epochs_a_3sec_18.fif'
save_name2_3s = 'epochs_b_3sec_18.fif'
epo1_c_3s, epo2_c_3s = ica_removal_short(epo1_3s, epo2_3s, ica1_3s, ica2_3s, exclude1_3s, exclude2_3s, save_name1_3s, save_name2_3s)