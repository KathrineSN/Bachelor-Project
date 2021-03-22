# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:50:23 2021

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

# Loading the data
def prepocess_long(file, plot=True):
    raw = mne.io.read_raw_bdf('Data\\' + file, preload=True)
    
    #raw.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
    
    #Filtering
    
    f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")
    
    # Dividing up into participants a and b
    
    picks_a = []
    picks_b = []
    channels = raw.info.ch_names
    
    for i in range(len(channels)):
        if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
            picks_a.append(channels[i])
    
    print(picks_a)
    
    for i in range(len(channels)):
        if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
            picks_b.append(channels[i])
    
    print(picks_b)
    
    # Epoching the data
    
    events = mne.find_events(f_raw, initial_event = True)
    print('Number of events:', len(events))
    print('Unique event codes:', np.unique(events[:, 2]))
    
    new_events = events[1::2,:]
    # Number of events
    
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
                  'Follower': 107, 'Control':108 } 
    
    epochs_a = mne.Epochs(f_raw, new_events, event_id = event_dict, tmin=-1.5, tmax=25,
                        baseline=(None, 0), picks = picks_a, preload=True, detrend = None)
    
    epochs_a.plot(n_epochs = 1, n_channels = 10)
    
    epochs_b = mne.Epochs(f_raw, new_events, event_id = event_dict, tmin=-1.5, tmax=25,
                        baseline=(None, 0), picks = picks_b, preload=True, detrend = None)
    
    epochs_b.plot(n_epochs = 1, n_channels = 10)
    
    # Downsampling to 256 Hz 
    
    epochs_a_resampled = epochs_a.copy().resample(256, npad = 'auto')
    epochs_b_resampled = epochs_b.copy().resample(256, npad = 'auto')
    
    epochs_a_resampled.plot(n_epochs = 1, n_channels = 10) 
    epochs_b_resampled.plot(n_epochs = 1, n_channels = 10) 
    
    # Plotting example of downsampling
    plt.figure(figsize=(7, 3))
    n_samples_to_plot = int(0.5 * epochs_a.info['sfreq'])  # plot 0.5 seconds of data
    plt.plot(epochs_a.times[:n_samples_to_plot],
             epochs_a.get_data()[0, 0, :n_samples_to_plot], color='black')
    
    n_samples_to_plot = int(0.5 * epochs_a_resampled.info['sfreq'])
    plt.plot(epochs_a_resampled.times[:n_samples_to_plot],
             epochs_a_resampled.get_data()[0, 0, :n_samples_to_plot],
             '-o', color='red')
    
    plt.xlabel('time (s)')
    plt.legend(['original', 'downsampled'], loc='best')
    plt.title('Effect of downsampling')
    mne.viz.tight_layout()
    
    # Saving resampled epochs
    #epochs_a_resampled.save('epochs_a_resampled-epo.fif', overwrite = True)
    #epochs_b_resampled.save('epochs_b_resampled-epo.fif', overwrite = True)
    
    # Loading epochs from fif
    #epochs_a_resampled = mne.read_epochs('epochs_a_resampled-epo.fif', preload = True)
    #epochs_b_resampled = mne.read_epochs('epochs_b_resampled-epo.fif', preload = True)
    
    # Setting correct channel names
    montage = mne.channels.make_standard_montage("biosemi64")
    new_ch_names = montage.ch_names
    
    for i in range(len(new_ch_names)):
        print(picks_a[i])
        print(new_ch_names[i])
        epochs_a_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})
    
    print(epochs_a_resampled.info.ch_names)
    
    
    for i in range(len(new_ch_names)):
        
        epochs_b_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_b_resampled.info.ch_names)
    
    # Renaming for use in HyPyP
    
    epo1 = epochs_a_resampled.copy()
    epo2 = epochs_b_resampled.copy()
    
    # Ensuring equal number of epochs
    
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    
    # Setting montage
    epo1.set_montage('biosemi64')
    epo2.set_montage('biosemi64')
    
    # Channel statistics
    df_a = epochs_a_resampled.to_data_frame(picks = new_ch_names)
    df_b = epochs_b_resampled.to_data_frame(picks = new_ch_names)
    ch_stat_a = df_a.describe()
    ch_stat_b = df_b.describe()
    if plot:
        #epochs_a.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        #epochs_b.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        epo1.plot_psd(fmin = 2, fmax = 40)
        epo2.plot_psd(fmin = 2, fmax = 40)
    
    return epo1, epo2

# Loading the data
def prepocess_short(file, plot=True):
    raw = mne.io.read_raw_bdf('Data\\' + file, preload=True)
    
    #raw.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
    
    #Filtering
    
    f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")
    
    # Dividing up into participants a and b
    
    picks_a = []
    picks_b = []
    channels = raw.info.ch_names
    
    for i in range(len(channels)):
        if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
            picks_a.append(channels[i])
    
    print(picks_a)
    
    for i in range(len(channels)):
        if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
            picks_b.append(channels[i])
    
    print(picks_b)
    
    # Epoching the data
    
    events = mne.find_events(f_raw, initial_event = True)
    print('Number of events:', len(events))
    print('Unique event codes:', np.unique(events[:, 2]))
    
    new_events = events[1::2,:]
    # Number of events
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
                  'Follower': 107, 'Control':108 } 
    
    # Creating new event list
    event_list = []       
    
    ev_list = np.zeros((26*len(new_events),3))
    #event_list.append(new_events[1,:])
    for i in range(len(new_events)):
        temp = np.reshape(np.tile(new_events[i,:],26),(-1,3))
        temp[0,0]-=1.5*2048
        temp[1:26,0] += np.arange(start=0, stop=25*2048, step=2048)
        ev_list[i*26:(i+1)*26] = temp
        event_list.append(temp)
        
    ev_list = ev_list.astype(int)

    fig = mne.viz.plot_events(ev_list, sfreq=raw.info['sfreq'],
                              first_samp=raw.first_samp)
    fig.subplots_adjust(right=0.7)
    
    epochs_a = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                        baseline=(None, None), picks = picks_a, preload=True, detrend = None)

    epochs_b = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                        baseline=(None, None), picks = picks_b, preload=True, detrend = None)
    
    epochs_a.plot(n_epochs = 25, n_channels = 10)


    # Downsampling to 256 Hz 
    epochs_a_resampled = epochs_a.copy().resample(256, npad = 'auto')
    epochs_b_resampled = epochs_b.copy().resample(256, npad = 'auto')
    
    epochs_a_resampled.plot(n_epochs = 25, n_channels = 10) 
    epochs_b_resampled.plot(n_epochs = 25, n_channels = 10) 
    
    # Plotting example of downsampling
    plt.figure(figsize=(7, 3))
    n_samples_to_plot = int(0.5 * epochs_a.info['sfreq'])  # plot 0.5 seconds of data
    plt.plot(epochs_a.times[:n_samples_to_plot],
             epochs_a.get_data()[0, 0, :n_samples_to_plot], color='black')
    
    n_samples_to_plot = int(0.5 * epochs_a_resampled.info['sfreq'])
    plt.plot(epochs_a_resampled.times[:n_samples_to_plot],
             epochs_a_resampled.get_data()[0, 0, :n_samples_to_plot],
             '-o', color='red')
    
    plt.xlabel('time (s)')
    plt.legend(['original', 'downsampled'], loc='best')
    plt.title('Effect of downsampling')
    mne.viz.tight_layout()
    
    # E.g. extracting uncoupled epochs
    
    #epochs_a_resampled['Uncoupled'].plot(n_epochs = 1, n_channels = 10) 
    
    # Saving resampled epochs
    #epochs_a_resampled.save('epochs_a_resampled-epo.fif', overwrite = True)
    #epochs_b_resampled.save('epochs_b_resampled-epo.fif', overwrite = True)
    
    # Loading epochs from fif
    #epochs_a_resampled = mne.read_epochs('epochs_a_resampled-epo.fif', preload = True)
    #epochs_b_resampled = mne.read_epochs('epochs_b_resampled-epo.fif', preload = True)
    
    # Setting correct channel names
    montage = mne.channels.make_standard_montage("biosemi64")
    #montage.plot()
    new_ch_names = montage.ch_names
    
    for i in range(len(new_ch_names)):
        print(picks_a[i])
        print(new_ch_names[i])
        epochs_a_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})
    
    print(epochs_a_resampled.info.ch_names)
    
    
    for i in range(len(new_ch_names)):
        
        epochs_b_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_b_resampled.info.ch_names)
    
    # Renaming for use in HyPyP
    
    
    #.save('epochs_a_resampled-epo.fif', overwrite = True)
    #epochs_b_resampled.save('epochs_b_resampled-epo.fif', overwrite = True)
    #epo1 = epochs_a_resampled['Coupled']
    #epo2 = epochs_b_resampled['Coupled']
    
    epo1 = epochs_a_resampled.copy()
    epo2 = epochs_b_resampled.copy()
    
    # Ensuring equal number of epochs
    
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    
    #biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    #biosemi_montage.plot(show_names=False)
    epo1.set_montage('biosemi64')
    epo2.set_montage('biosemi64')
    
    # Channel statistics
    df_a = epochs_a_resampled.to_data_frame(picks = new_ch_names)
    df_b = epochs_b_resampled.to_data_frame(picks = new_ch_names)
    ch_stat_a = df_a.describe()
    ch_stat_b = df_b.describe()
    if plot:
        #epochs_a.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        #epochs_b.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        epo1.plot_psd(fmin = 2, fmax = 40)
        epo2.plot_psd(fmin = 2, fmax = 40)
    
    return epo1, epo2

def bad_removal(epo, bads):
    for bad in bads:
        epo.info['bads'].append(bad)
        
        return epo
    
def ica_part(epo):
    
    ica1 = mne.preprocessing.ICA(n_components=15,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)
    
    ica1.fit(epo)
   
    ica1.plot_components()
    
    ica1.plot_sources(epo, show_scrollbars=True)
    return ica1
    
def ica_removal_long(epo,ica1, exclude_list, save_name):
    epo_cleaned = ica1.apply(epo, exclude = exclude_list)
    epo_cleaned.interpolate_bads()
    epo_cleaned.set_eeg_reference('average')
    epo_cleaned.save(save_name, overwrite = True)
    
    return epo_cleaned

def ica_removal_short(epo1, epo2, ica1, ica2, exclude_list1, exclude_list2, save_name1, save_name2):
    epo_cleaned1 = ica1.apply(epo1, exclude = exclude_list1)
    epo_cleaned2 = ica2.apply(epo2, exclude = exclude_list2)
    epo_cleaned1.interpolate_bads()
    epo_cleaned2.interpolate_bads()
    cleaned_epochs_AR, dic_AR = prep.AR_local([epo_cleaned1, epo_cleaned2],
                                          strategy="union",
                                          threshold=50.0,
                                          verbose=True)
    epo_cleaned1.set_eeg_reference('average')
    epo_cleaned2.set_eeg_reference('average')
    epo_cleaned1.save(save_name1, overwrite = True)
    epo_cleaned2.save(save_name2, overwrite = True)
    
    return epo_cleaned1, epo_cleaned2   

def ccorr(epochs_a, epochs_b, pair_name, length, drop_list):
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 }
    
    conditions = ['Coupled', 'Uncoupled', 'Leader', 'Control']
    
    if length == 'long':
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
            
            result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = True)
            
            #Get inter brain part of the matrix
            n_ch = len(epochs_a.info['ch_names'])
            theta, alpha, beta = result[:, 0:n_ch, n_ch:2*n_ch]
            
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
            Epoch_no = drop_list
            if i in Epoch_no:
                #print(i)
                epo_drop.remove(i)
        
        # Ensuring list is no longer than the number of epochs     
        while epo_drop[-1]>(len(epochs_b)-1):
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
        
        #equalize number of epochs used to calculate connectivity values
        #mne.epochs.equalize_epoch_counts([epo_a, epo_b])
        
        for c in conditions:
               # Merging the leader and follower
            if c == 'Leader':
                epo_a_l = epochs_a['Leader']
                epo_b_l = epochs_b['Leader']
                epo_a_f = epochs_a['Follower']
                epo_b_f = epochs_b['Follower']
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
                
                result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average = False)
                
                #Defining the number of channels
                n_ch = len(epochs_a.info['ch_names'])
                
                #Averaging over the epochs specific to the given trial
                trials = []
                
                for j in range(3):
                    for i in d[event_dict['Leader']] or d[event_dict['Follower']]:
                        trials.append(sum(result[j,0:i,:,:])/i)
                
                '''
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
                '''
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
                
            else: 
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
                
                '''
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
                '''
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

