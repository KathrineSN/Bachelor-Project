# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:17:32 2021

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


path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)

epochs_a = mne.read_epochs('epochs_a_preprocessed-epo.fif', preload = True)
epochs_b = mne.read_epochs('epochs_b_preprocessed-epo.fif', preload = True)
epochs_a_s = mne.read_epochs('epochs_a_short.fif', preload = True)
epochs_b_s = mne.read_epochs('epochs_b_short.fif', preload = True)

def ccorr(epochs_a, epochs_b, pair_name, length):
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 }
    
    conditions = ['Coupled', 'Uncoupled', 'Leader', 'Follower', 'Control']
    
    if length == 'long':
        epochs_a.crop(tmin = 2, tmax = 23)
        epochs_a.plot(n_epochs = 1, n_channels = 10)
        epochs_b.crop(tmin = 2, tmax = 23)
    
        for i in conditions:
            print(i)
            epo_a = epochs_a[i]
            epo_b = epochs_b[i]
            
            #Defining frequency bands
            freq_bands = {'Theta': [4, 7],
                          'Alpha' :[8, 13],
                          'Beta': [15, 25]}
            
            freq_bands = OrderedDict(freq_bands)
            
            sampling_rate = epo_a.info['sfreq']
            
            #Connectivity
            
            #Data and storage
            data_inter = np.array([epo_a, epo_b])
            #result_intra = []
            
            #Analytic signal per frequency band
            complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                         freq_bands)
            
            result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = True)
            
            #Get inter brain part of the matrix
            n_ch = len(epochs_a.info['ch_names'])
            theta, alpha, beta = result[:, 0:n_ch, n_ch:2*n_ch]
            
            theta = abs(theta - np.mean(theta[:]) / np.std(theta[:]))
            alpha = abs(alpha - np.mean(alpha[:]) / np.std(alpha[:]))
            beta = abs(beta - np.mean(beta[:]) / np.std(beta[:]))
            
            print('Range of the connectivities:')
            print('Theta max:' + str(np.max(theta)))
            print('Theta min:' + str(np.min(theta)))
            print('Alpha max:' + str(np.max(alpha)))
            print('Alpha min:' + str(np.min(alpha)))
            print('Beta max:' + str(np.max(beta)))
            print('Beta min:' + str(np.min(beta)))
            
            np.save(pair_name + '_theta_' + i + '_' + length, theta)
            np.save(pair_name + '_alpha_' + i + '_' + length, alpha)
            np.save(pair_name + '_beta_' + i + '_' + length, beta)
    
    if length == 'short':
        
        #conditions = ['Coupled', 'Uncoupled', 'Leader', 'Follower', 'Control']
        
        epo_drop = []
        epo_drop.append(0)
        epo_drop.append(1)
        epo_drop.append(2)
        epo_drop.append(24)
        epo_drop.append(25)
        for i in range(64*5):
            epo_drop.append(epo_drop[i]+26)
        
        # Ensuring that already removed epochs are not in list
        for i in epo_drop:
            Epoch_no = [91, 108, 300, 301, 341, 354, 355, 356, 381, 382, 383, 397, 398, 416, 442, 443, 473, 474, 476, 477, 497, 498, 502, 507, 508, 509, 510, 511, 512, 513, 514, 528, 530, 550, 551, 553, 554, 555, 556, 557, 559, 561, 578, 585, 586, 587, 588, 589, 603, 604, 605, 622, 632, 654, 658, 660, 663, 664, 675, 677, 678, 683, 684, 686, 720, 721, 723, 724, 725, 727, 908, 967, 975, 976, 993, 1027, 1031, 1033, 1034, 1036, 1041, 1042, 1072, 1073, 1125, 1126, 1145, 1146, 1147, 1148, 1279, 1292, 1303, 1314, 1315, 1329, 1338, 1339, 1340, 1347, 1349, 1351, 1358, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1378, 1395, 1396, 1408, 1439, 1463, 1486, 1514, 1515, 1517, 1519, 1520, 1521, 1522, 1523, 1628, 1631, 1632, 1633, 1643, 1655, 1656, 1657, 1660, 1661, 1662, 1663]
            if i in Epoch_no:
                #print(i)
                epo_drop.remove(i)
        
        # Ensuring list is no longer than the number of epochs     
        while epo_drop[-1]>(len(epochs_a)-1):
            epo_drop.pop(-1)
          
        # Dropping the beginning and end of a trial      
        epo_a = epochs_a.drop(epo_drop)
        epo_b = epochs_b.drop(epo_drop)
        
        # Getting the number of epochs of specific condition in a row
        
        a = epo_a.events[:,2]
        d = dict()
        
        for k, v in groupby(a):
            d.setdefault(k, []).append(len(list(v)))
        #print(d)
        
        for c in conditions:
            epo_a_c = epo_a[c]
            epo_b_c = epo_b[c]
            
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
            
            result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = False)
            
            #Defining the number of channels
            n_ch = len(epochs_a.info['ch_names'])
            
            #Averaging over the epochs specific to the given trial
            trials = []
            
            for j in range(3):
                for i in d[event_dict[c]]:
                    trials.append(sum(result[j,0:i,:,:])/i)
            
            if c == 'Leader' or c == 'Follower':
                print('LF')
                print(len(trials))
                theta = sum(trials[::3])/8
                alpha = sum(trials[1::3])/8
                beta = sum(trials[2::3])/8
            
            else:
                theta = sum(trials[::3])/16
                alpha = sum(trials[1::3])/16
                beta = sum(trials[2::3])/16
            
            theta = theta[0:n_ch, n_ch:2*n_ch]
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
            
            np.save(pair_name + '_theta_' + c + '_' + length, theta)
            np.save(pair_name + '_alpha_' + c + '_' + length, alpha)
            np.save(pair_name + '_beta_' + c + '_' + length, beta)
        
      
    return theta, alpha, beta

ccorr(epochs_a, epochs_b, 'pair003', 'long')
ccorr(epochs_a_s, epochs_b_s, 'pair003', 'short')

    
epochs_a_s.plot(n_channels = 10, n_epochs = 10)

# Creating list of beginning epochs to remove

conditions = ['Coupled', 'Uncoupled', 'Leader', 'Follower', 'Control']
epo_drop = []
epo_drop.append(0)
epo_drop.append(1)
epo_drop.append(2)
epo_drop.append(24)
epo_drop.append(25)
for i in range(64*5):
    epo_drop.append(epo_drop[i]+26)

# Ensuring that already removed epochs are not in list
for i in epo_drop:
    Epoch_no = [91, 108, 300, 301, 341, 354, 355, 356, 381, 382, 383, 397, 398, 416, 442, 443, 473, 474, 476, 477, 497, 498, 502, 507, 508, 509, 510, 511, 512, 513, 514, 528, 530, 550, 551, 553, 554, 555, 556, 557, 559, 561, 578, 585, 586, 587, 588, 589, 603, 604, 605, 622, 632, 654, 658, 660, 663, 664, 675, 677, 678, 683, 684, 686, 720, 721, 723, 724, 725, 727, 908, 967, 975, 976, 993, 1027, 1031, 1033, 1034, 1036, 1041, 1042, 1072, 1073, 1125, 1126, 1145, 1146, 1147, 1148, 1279, 1292, 1303, 1314, 1315, 1329, 1338, 1339, 1340, 1347, 1349, 1351, 1358, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1378, 1395, 1396, 1408, 1439, 1463, 1486, 1514, 1515, 1517, 1519, 1520, 1521, 1522, 1523, 1628, 1631, 1632, 1633, 1643, 1655, 1656, 1657, 1660, 1661, 1662, 1663]
    if i in Epoch_no:
        print(i)
        epo_drop.remove(i)

# Ensuring list is no longer than the number of epochs     
while epo_drop[-1]>1356:
    print(epo_drop.pop(-1))
  
# Dropping the beginning and end of a trial      
epo_a = epochs_a_s.drop(epo_drop)
epo_b = epochs_b_s.drop(epo_drop)

# Getting the number of epochs of specific condition in a row
from itertools import groupby

a = epo_a.events[:,2]
d = dict()

for k, v in groupby(a):
    d.setdefault(k, []).append(len(list(v)))
print(d)

#for i in conditions:
    #print(i)
epo_a = epo_a['Coupled']
epo_b = epo_b['Coupled']

#Defining frequency bands
freq_bands = {'Theta': [4, 7],
              'Alpha' :[8, 13],
              'Beta': [15, 25]}

freq_bands = OrderedDict(freq_bands)

sampling_rate = epo_a.info['sfreq']

#Connectivity

#Data and storage
data_inter = np.array([epo_a, epo_b])
#result_intra = []

#Analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                             freq_bands)



result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = False)


#Get inter brain part of the matrix
n_ch = len(epochs_a.info['ch_names'])
#theta, alpha, beta = result[:,:, 0:n_ch, n_ch:2*n_ch]

#Averaging over the epochs specific to the given trial
trials = []

for j in range(3):
    for i in d[103]:
        trials.append(sum(result[j,0:i,:,:])/i)

theta = sum(trials[::3])/16
alpha = sum(trials[1::3])/16
beta = sum(trials[2::3])/16

theta = theta[0:n_ch, n_ch:2*n_ch]
alpha = alpha[0:n_ch, n_ch:2*n_ch]
beta = beta[0:n_ch, n_ch:2*n_ch]

theta = abs(theta - np.mean(theta[:]) / np.std(theta[:]))
alpha = abs(alpha - np.mean(alpha[:]) / np.std(alpha[:]))
beta = abs(beta - np.mean(beta[:]) / np.std(beta[:]))

print('Range of the connectivities:')
print('Theta max:' + str(np.max(theta)))
print('Theta min:' + str(np.min(theta)))
print('Alpha max:' + str(np.max(alpha)))
print('Alpha min:' + str(np.min(alpha)))
print('Beta max:' + str(np.max(beta)))
print('Beta min:' + str(np.min(beta)))
    


# Alpha low for example
#values = theta

# Subtracts the diagonal
#values -= np.diag(np.diag(values))

# Normalising connectivity values
#C = (values - np.mean(values[:])) / np.std(values[:])

#Slicing results to get the intra-brain part of matrix
#for i in [0, 1]:
#    theta, alpha, beta = result[:, i:i+n_ch, i:i+n_ch]
    # choosing Alpha_Low for futher analyses for example
#    values_intra = alpha
#    values_intra -= np.diag(np.diag(values_intra))
    # computing Cohens'D for further analyses for example
#    C_intra = (values_intra -
#              np.mean(values_intra[:])) / np.std(values_intra[:])
    # can also sample CSD values directly for statistical analyses
#    result_intra.append(C_intra)


### Comparing inter-brain connectivity values to random signal
#data = [np.array([values, values]), np.array([result_intra[0], result_intra[0]])]

#statscondCluster = stats.statscondCluster(data=data,
#                                          freqs_mean=np.arange(7.5, 11),
#                                          ch_con_freq=None,
#                                          tail=0,
#                                          n_permutations=5000,
#                                         alpha=0.05)

#Visualization of interbrain connectivity in 2D
#viz.viz_2D_topomap_inter(epochs_a, epochs_b, C, threshold='auto', steps=10, lab=True)

#Visualization of interbrain connectivity in 3D
#viz.viz_3D_inter(epochs_a, epochs_b, C, threshold='auto', steps=10, lab=False)








