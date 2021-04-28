# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:14:59 2021

@author: kathr
"""
import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hypyp import prep 
from hypyp import analyses
from hypyp import stats
from hypyp import viz
from collections import Counter
from collections import OrderedDict
from itertools import groupby
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)

#%%
epochs_a_3sec = mne.read_epochs('epochs_a_6sec_3.fif')
epochs_b_3sec = mne.read_epochs('epochs_b_6sec_3.fif')


conditions = ['Coupled', 'Uncoupled', 'Leader', 'Control']
epo_drop = []
epo_drop.append(0)
epo_drop.append(4)
for i in range(63*2): #Previously was 64*5 changed as first trial is already appended
    epo_drop.append(epo_drop[i]+5)
print(len(epo_drop))

# Dropping the beginning and end of a trial      
epo_a = epochs_a_3sec.drop(epo_drop)        
epo_b = epochs_b_3sec.drop(epo_drop) 

epo_a_copy = epo_a.copy()
epo_b_copy = epo_b.copy()


# Running autoreject function
cleaned_epochs_AR, dic_AR = prep.AR_local([epo_a_copy, epo_b_copy],
                                strategy="union",
                                threshold=50.0,
                                verbose=True)

epo_a_cleaned = cleaned_epochs_AR[0]
epo_b_cleaned = cleaned_epochs_AR[1]

# Getting the number of epochs of specific condition in a row

a = epo_a_copy.events[:,2]
d = dict()

for k, v in groupby(a):
    d.setdefault(k, []).append(len(list(v)))

#%%
event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 }
for c in conditions:
       # Merging the leader and follower
    if c == 'Leader':
        epo_a_l = epo_a_cleaned['Leader']
        epo_b_l = epo_b_cleaned['Leader']
        epo_a_f = epo_a_cleaned['Follower']
        epo_b_f = epo_b_cleaned['Follower']
        epo_a_c = mne.concatenate_epochs([epo_a_l, epo_b_f])
        epo_b_c = mne.concatenate_epochs([epo_b_l, epo_a_f])
        c = 'Leader-Follower'
        
        freq_bands = {'Theta': [4, 7],
                  'Alpha' :[8, 13],
                  'Beta': [15, 25]}
    
        freq_bands = OrderedDict(freq_bands)
        
        sampling_rate = epo_a_c.info['sfreq']
        
        #Connectivity
        
        #Data and storage
        data_inter = np.array([epo_a_c, epo_b_c])
        
        #Analytic signal per frequency band
        complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                     freq_bands)
        
        result, angle, _,_ = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = False)
        
        #Defining the number of channels
        n_ch = len(epo_a_cleaned.info['ch_names'])
        
        #Averaging over the epochs specific to the given trial
        trials = []
        
        for j in range(3):
            for i in d[event_dict['Leader']] + d[event_dict['Follower']]:
                trials.append(sum(result[j,0:i,:,:])/i)
        
        
        theta = sum(trials[::3])/len(trials[::3])
        alpha = sum(trials[1::3])/len(trials[::3])
        beta = sum(trials[2::3])/len(trials[::3])
            
        theta = theta[0:n_ch, n_ch:2*n_ch]
        alpha = alpha[0:n_ch, n_ch:2*n_ch]
        beta = beta[0:n_ch, n_ch:2*n_ch]
        
        #theta = abs(theta - np.mean(theta[:]) / np.std(theta[:]))
        #alpha = abs(alpha - np.mean(alpha[:]) / np.std(alpha[:]))
        #beta = abs(beta - np.mean(beta[:]) / np.std(beta[:]))
        
        print(c)
        print('Range of the connectivities:')
        print('Theta max:' + str(np.max(theta)))
        print('Theta min:' + str(np.min(theta)))
        print('Alpha max:' + str(np.max(alpha)))
        print('Alpha min:' + str(np.min(alpha)))
        print('Beta max:' + str(np.max(beta)))
        print('Beta min:' + str(np.min(beta)))
        
        np.save('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_theta_' + c + '_3sec', theta)
        np.save('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_alpha_' + c + '_3sec', alpha)
        np.save('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_beta_' + c + '_3sec', beta)
        #np.save('Connectivity matrices/lengths/6sec/coh/coh_pair003_theta_' + c + '_3sec', theta)
        #np.save('Connectivity matrices/lengths/6sec/coh/coh_pair003_alpha_' + c + '_3sec', alpha)
        #np.save('Connectivity matrices/lengths/6sec/coh/coh_pair003_beta_' + c + '_3sec', beta)
        
        
    else: 
        epo_a_c = epo_a_cleaned[c]
        epo_b_c = epo_b_cleaned[c]
        
        print('no. of epochs')
        len(epo_a_c)
        #Defining frequency bands
        freq_bands = {'Theta': [4, 7],
                      'Alpha' :[8, 13],
                      'Beta': [15, 25]}
        
        freq_bands = OrderedDict(freq_bands)
        
        sampling_rate = epo_a_c.info['sfreq']
        
        #Connectivity
        
        #Data and storage
        data_inter = np.array([epo_a_c, epo_b_c])
        
        #Analytic signal per frequency band
        complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                     freq_bands)
        
        result, angle, _,_ = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = False)
        
        #Defining the number of channels
        n_ch = len(epo_a_cleaned.info['ch_names'])
        
        #Averaging over the epochs specific to the given trial
        trials = []
        
        for j in range(3):
            for i in d[event_dict[c]]:
                trials.append(sum(result[j,0:i,:,:])/i)
        
        print(c)
        print(len(trials[::3]))
        theta = sum(trials[::3])/len(trials[::3])
        alpha = sum(trials[1::3])/len(trials[::3])
        beta = sum(trials[2::3])/len(trials[::3])
            
        theta = theta[0:n_ch, n_ch:2*n_ch] # Skal det her være her?? Bliver den ikke allerede sliced i for-loopet??
        alpha = alpha[0:n_ch, n_ch:2*n_ch]
        beta = beta[0:n_ch, n_ch:2*n_ch]
        
        theta = abs(theta - np.mean(theta[:]) / np.std(theta[:]))
        alpha = abs(alpha - np.mean(alpha[:]) / np.std(alpha[:]))
        beta = abs(beta - np.mean(beta[:]) / np.std(beta[:]))
        
        print(c)
        print('Range of the connectivities:')
        print('Theta max:' + str(np.max(theta)))
        print('Theta min:' + str(np.min(theta)))
        print('Alpha max:' + str(np.max(alpha)))
        print('Alpha min:' + str(np.min(alpha)))
        print('Beta max:' + str(np.max(beta)))
        print('Beta min:' + str(np.min(beta)))
        
        np.save('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_theta_' + c + '_3sec', theta)
        np.save('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_alpha_' + c + '_3sec', alpha)
        np.save('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_beta_' + c + '_3sec', beta)
        
#%% Creating matrices for coupled
coupled_1sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Coupled_short.npy')
fig = plt.figure()
plt.title('ccorr coupled 1 sec')
plt.imshow(coupled_1sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()


coupled_1sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Coupled_short.npy')
fig = plt.figure()
plt.title('coh coupled 1 sec')
plt.imshow(coupled_1sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 


coupled_3sec_ccorr = np.load('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_alpha_Coupled_3sec.npy')

fig = plt.figure()
plt.title('ccorr coupled 3 sec')
plt.imshow(coupled_3sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()        
 
coupled_3sec_coh = np.load('Connectivity matrices/lengths/3sec/coh/coh_pair003_alpha_Coupled_3sec.npy')

fig = plt.figure()
plt.title('coh coupled 3 sec')
plt.imshow(coupled_3sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()  

coupled_6sec_ccorr = np.load('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_alpha_Coupled_3sec.npy')
fig = plt.figure()
plt.title('ccorr coupled 6 sec')
plt.imshow(coupled_6sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 
  
coupled_6sec_coh = np.load('Connectivity matrices/lengths/6sec/coh/coh_pair003_alpha_Coupled_3sec.npy')
fig = plt.figure()
plt.title('coh coupled 6 sec')
plt.imshow(coupled_6sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 

coupled_25sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Coupled_long.npy')
fig = plt.figure()
plt.title('ccorr coupled 25 sec')
plt.imshow(coupled_25sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()


coupled_25sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Coupled_long.npy')
fig = plt.figure()
plt.title('coh coupled 25 sec')
plt.imshow(coupled_25sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 

#%% Uncoupled
uncoupled_1sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Uncoupled_short.npy')
fig = plt.figure()
plt.title('ccorr uncoupled 1 sec')
plt.imshow(uncoupled_1sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()


uncoupled_1sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Uncoupled_short.npy')
fig = plt.figure()
plt.title('coh uncoupled 1 sec')
plt.imshow(uncoupled_1sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 


uncoupled_3sec_ccorr = np.load('Connectivity matrices/lengths/3sec/ccorr/ccorr_pair003_alpha_Uncoupled_3sec.npy')

fig = plt.figure()
plt.title('ccorr uncoupled 3 sec')
plt.imshow(uncoupled_3sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()        
 
uncoupled_3sec_coh = np.load('Connectivity matrices/lengths/3sec/coh/coh_pair003_alpha_Uncoupled_3sec.npy')

fig = plt.figure()
plt.title('coh uncoupled 3 sec')
plt.imshow(uncoupled_3sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()  

uncoupled_6sec_ccorr = np.load('Connectivity matrices/lengths/6sec/ccorr/ccorr_pair003_alpha_Uncoupled_3sec.npy')
fig = plt.figure()
plt.title('ccorr uncoupled 6 sec')
plt.imshow(uncoupled_6sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 
  
uncoupled_6sec_coh = np.load('Connectivity matrices/lengths/6sec/coh/coh_pair003_alpha_Uncoupled_3sec.npy')
fig = plt.figure()
plt.title('coh uncoupled 6 sec')
plt.imshow(uncoupled_6sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 

uncoupled_25sec_ccorr = np.load('Connectivity matrices/ccorr/ccorr_pair003_alpha_Uncoupled_long.npy')
fig = plt.figure()
plt.title('ccorr uncoupled 25 sec')
plt.imshow(uncoupled_25sec_ccorr,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show()


uncoupled_25sec_coh = np.load('Connectivity matrices/coh/coh_pair003_alpha_Uncoupled_long.npy')
fig = plt.figure()
plt.title('coh uncoupled 25 sec')
plt.imshow(uncoupled_25sec_coh,cmap=plt.cm.Reds)
plt.clim(0,0.5)
plt.colorbar()
plt.show() 

#%% contrasts
# 1 sec
contrast_1sec_ccorr = coupled_1sec_ccorr-uncoupled_1sec_ccorr
fig = plt.figure()
plt.title('ccorr coupled-uncoupled 1 sec')
plt.imshow(contrast_1sec_ccorr,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show() 

contrast_1sec_coh = coupled_1sec_coh-uncoupled_1sec_coh
fig = plt.figure()
plt.title('coh coupled-uncoupled 1 sec')
plt.imshow(contrast_1sec_coh,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show()

# 3 sec
contrast_3sec_ccorr = coupled_3sec_ccorr-uncoupled_3sec_ccorr
fig = plt.figure()
plt.title('ccorr coupled-uncoupled 3 sec')
plt.imshow(contrast_3sec_ccorr,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show() 

contrast_3sec_coh = coupled_3sec_coh-uncoupled_3sec_coh
fig = plt.figure()
plt.title('coh coupled-uncoupled 3 sec')
plt.imshow(contrast_3sec_coh,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show()  

# 6 sec
contrast_6sec_ccorr = coupled_6sec_ccorr-uncoupled_6sec_ccorr
fig = plt.figure()
plt.title('ccorr coupled-uncoupled 6 sec')
plt.imshow(contrast_6sec_ccorr,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show() 

contrast_6sec_coh = coupled_6sec_coh-uncoupled_6sec_coh
fig = plt.figure()
plt.title('coh coupled-uncoupled 6 sec')
plt.imshow(contrast_6sec_coh,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show()  

# 25 sec
contrast_25sec_ccorr = coupled_25sec_ccorr-uncoupled_25sec_ccorr
fig = plt.figure()
plt.title('ccorr coupled-uncoupled 25 sec')
plt.imshow(contrast_25sec_ccorr,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show() 

contrast_25sec_coh = coupled_25sec_coh-uncoupled_25sec_coh
fig = plt.figure()
plt.title('coh coupled-uncoupled 25 sec')
plt.imshow(contrast_25sec_coh,cmap=plt.cm.seismic)
plt.clim(-0.3,0.3)
plt.colorbar()
plt.show()  






        