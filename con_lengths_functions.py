# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:00:36 2021

@author: kathr
"""
import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from hypyp import prep 
from hypyp import analyses
from collections import OrderedDict
from itertools import groupby

def ccorr(epochs_a, epochs_b, pair_name, length, drop_list):
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 }
    
    conditions = ['Coupled', 'Uncoupled', 'Leader', 'Control']
    
    if length == 'twelve sec':
        epochs_a.crop(tmin = 2, tmax = 23)
        epochs_a.plot(n_epochs = 1, n_channels = 10)
        epochs_b.crop(tmin = 2, tmax = 23)
    
        for i in conditions:
            
            # Merging the leader and follower
            if i == 'Leader':
                epo_a_l = epochs_a['Leader']
                epo_b_l = epochs_b['Leader']
                epo_a_f = epochs_a['Follower']
                epo_b_f = epochs_b['Follower']
                epo_a = mne.concatenate_epochs([epo_a_l, epo_b_f])
                epo_b = mne.concatenate_epochs([epo_b_l, epo_a_f])
                i = 'Leader-Follower'
            else: 
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
            
            result, angle,_,_ = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = True)
            
            #Get inter brain part of the matrix
            n_ch = len(epochs_a.info['ch_names'])
            #result = result[0]
            theta, alpha, beta = result[:, 0:n_ch, n_ch:2*n_ch]
            '''
            plt.figure()
            plt.imshow(theta,cmap=plt.cm.hot)
            plt.clim(0,0.8)
            plt.colorbar()
            plt.show()
            
            plt.figure()
            plt.imshow(alpha,cmap=plt.cm.hot)
            plt.clim(0,0.8)
            plt.colorbar()
            plt.show()
            
            plt.figure()
            plt.imshow(beta,cmap=plt.cm.hot)
            plt.clim(0,0.8)
            plt.colorbar()
            plt.show()
            '''
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
            
            np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_theta_' + i + '_' + length, theta)
            np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_alpha_' + i + '_' + length, alpha)
            np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_beta_' + i + '_' + length, beta)
    
    if length == 'short':
        
        #conditions = ['Coupled', 'Uncoupled', 'Leader', 'Follower', 'Control']
        epo_drop = []
        epo_drop.append(0)
        epo_drop.append(1)
        epo_drop.append(2)
        epo_drop.append(24)
        epo_drop.append(25)
        for i in range(63*5): #Previously was 64*5 changed as first trial is already appended
            epo_drop.append(epo_drop[i]+26)
        print(len(epo_drop))
        '''
        # Ensuring that already removed epochs are not in list
        for i in epo_drop:
            Epoch_no = drop_list
            if i in Epoch_no:
                #print(i)
                epo_drop.remove(i)
        
        # Ensuring list is no longer than the number of epochs     
        while epo_drop[-1]>(len(epochs_b)-1):
            epo_drop.pop(-1)
        '''  
        # Dropping the beginning and end of a trial      
        epo_a = epochs_a.drop(epo_drop)        
        epo_b = epochs_b.drop(epo_drop)
        
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
        #print(d)
        
        #equalize number of epochs used to calculate connectivity values
        #mne.epochs.equalize_epoch_counts([epo_a, epo_b])
        
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
                n_ch = len(epochs_a.info['ch_names'])
                
                #Averaging over the epochs specific to the given trial
                trials = []
                
                for j in range(3):
                    for i in d[event_dict['Leader']] + d[event_dict['Follower']]:
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
                
                print(c)
                print('Range of the connectivities:')
                print('Theta max:' + str(np.max(theta)))
                print('Theta min:' + str(np.min(theta)))
                print('Alpha max:' + str(np.max(alpha)))
                print('Alpha min:' + str(np.min(alpha)))
                print('Beta max:' + str(np.max(beta)))
                print('Beta min:' + str(np.min(beta)))
                
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_theta_' + c + '_' + length, theta)
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_alpha_' + c + '_' + length, alpha)
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_beta_' + c + '_' + length, beta)
                
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
                n_ch = len(epochs_a.info['ch_names'])
                
                #Averaging over the epochs specific to the given trial
                trials = []
                
                for j in range(3):
                    for i in d[event_dict[c]]:
                        trials.append(sum(result[j,0:i,:,:])/i)
                '''
                if c == 'Resting':
                    print('Resting')
                    print(len(trials[::3]))
                    theta = sum(trials[::3])/2
                    alpha = sum(trials[1::3])/2
                    beta = sum(trials[2::3])/2
                
                else:
                    print(c)
                    print(len(trials[::3]))
                    theta = sum(trials[::3])/16
                    alpha = sum(trials[1::3])/16
                    beta = sum(trials[2::3])/16
                '''
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
                
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_theta_' + c + '_' + length, theta)
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_alpha_' + c + '_' + length, alpha)
                np.save('Connectivity matrices/ccorr/' + 'ccorr_' + pair_name + '_beta_' + c + '_' + length, beta)
        
      
    return theta, alpha, beta, angle, complex_signal, epo_a_cleaned, epo_a