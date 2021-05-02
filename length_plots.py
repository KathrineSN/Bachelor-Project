# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:43:54 2021

@author: kathr
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from con_functions import load_avg_matrix

path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)

#%% Length data for one pair

coupled_1sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Coupled_short.npy')

coupled_1sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Coupled_short.npy')

coupled_3sec_ccorr = np.load('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_alpha_Coupled_3sec.npy')
        
coupled_3sec_coh = np.load('Connectivity matrices/lengths/3sec/coh/coh_pair003_alpha_Coupled_3sec.npy')  

coupled_6sec_ccorr = np.load('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_alpha_Coupled_3sec.npy')
  
coupled_6sec_coh = np.load('Connectivity matrices/lengths/6sec/coh/coh_pair003_alpha_Coupled_3sec.npy')

coupled_25sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Coupled_long.npy')

coupled_25sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Coupled_long.npy')

#%% Average data
ccorr_alpha_cou_s = load_avg_matrix('ccorr','alpha', 'Coupled', 'short', save = 0)
ccorr_alpha_cou_3s = load_avg_matrix('ccorr','alpha', 'Coupled', '3sec', save = 0)
ccorr_alpha_cou = load_avg_matrix('ccorr','alpha', 'Coupled', 'long', save = 0)

coh_alpha_cou_s = load_avg_matrix('coh','alpha', 'Coupled', 'short', save = 0)
coh_alpha_cou_3s = load_avg_matrix('coh','alpha', 'Coupled', '3sec', save = 0)
coh_alpha_cou = load_avg_matrix('coh','alpha', 'Coupled', 'long', save = 0)


#%% Bar plot for lengths for one pair

coherence_values = np.array([np.mean(coupled_1sec_coh), np.mean(coupled_3sec_coh),np.mean(coupled_6sec_coh),np.mean(coupled_25sec_coh)])
ccorr_values = np.array([np.mean(coupled_1sec_ccorr), np.mean(coupled_3sec_ccorr),np.mean(coupled_6sec_ccorr),np.mean(coupled_25sec_ccorr)])
epo_length = np.array([1,3,6,25])

plt.figure(figsize=(10, 3))
plt.suptitle('Connectivity value variability for one pair')
plt.subplot(1,2,1)
plt.xlabel('epoch length')
plt.ylabel('average coherence value')
plt.plot(epo_length,coherence_values)
plt.show()

plt.subplot(1,2,2)
plt.xlabel('epoch length')
plt.ylabel('average circular correlation value')
plt.plot(epo_length,ccorr_values)
plt.show()

#%% Bar plot with average of all pairs

coherence_values = np.array([np.mean(coh_alpha_cou_s), np.mean(coh_alpha_cou_3s),np.mean(coh_alpha_cou)])
ccorr_values = np.array([np.mean(ccorr_alpha_cou_s), np.mean(ccorr_alpha_cou_3s),np.mean(ccorr_alpha_cou)])
epo_length = np.array([1,3,25])

plt.figure(figsize=(10, 3))
plt.suptitle('Connectivity value variability for all pairs')
plt.subplot(1,2,1)
plt.xlabel('epoch length')
plt.ylabel('average coherence value')
plt.plot(epo_length,coherence_values)
plt.show()

plt.subplot(1,2,2)
plt.xlabel('epoch length')
plt.ylabel('average circular correlation value')
plt.plot(epo_length,ccorr_values)
plt.show()