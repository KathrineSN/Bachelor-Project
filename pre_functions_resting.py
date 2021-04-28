
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
    
    
    event_dict = {'Resting': 101,'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
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
    
    event_dict = {'Resting': 101, 'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
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

def ICA_result(df_afterICA, df_beforeICA, start, length, chan_start, chan_end):
    
    montage = mne.channels.make_standard_montage("biosemi64")
    new_ch_names = montage.ch_names
    
    signals_after = []
    signals_before = []
    for i in range(len(new_ch_names)):
        
        df_afterICA = df_afterICA.iloc[start:]
        df_beforeICA = df_beforeICA.iloc[start:]

        #t = df_afterICA.loc[(df_afterICA['time'] >= start) & (df_afterICA['time'] <= stop)]
        t = df_afterICA[['time']].head(length)
        #signals_after.append(df_afterICA.loc[(df_afterICA[new_ch_names[i]] >= start) & (df_afterICA[new_ch_names[i]] <= stop)])
        #signals_before.append(df_beforeICA.loc[(df_beforeICA[new_ch_names[i]] >= start) & (df_beforeICA[new_ch_names[i]] <= stop)])

        signals_after.append(df_afterICA[[new_ch_names[i]]].head(length))
        signals_before.append(df_beforeICA[[new_ch_names[i]]].head(length))

    print(df_afterICA)
    #print(t[len(t)-1])
    
    fig = plt.figure()
    no_chan = chan_end-chan_start
    for i in range(no_chan):
        
        #temp = i
        ax = plt.subplot(no_chan,1,i+1)
        plt.plot(t ,signals_before[chan_start + i], t, signals_after[chan_start + i])
        plt.subplots_adjust(hspace = .001)

        ax.title.set_visible(False)
        ax.set_ylabel(new_ch_names[chan_start + i])
        ax.set_xlabel('time (ms)')
        
    
    #plt.set_xlabel('time')
    return plt.show()

def before_vs_after_ICA(epochs_a, epochs_b, epochs_a_cleaned, epochs_b_cleaned):
    df_a = epochs_a.to_data_frame()
    df_b = epochs_b.to_data_frame()
    
    df_clean_a = epochs_a_cleaned.to_data_frame()
    df_clean_b = epochs_b_cleaned.to_data_frame()
    
    #Extracting the 5th coupled epoch
    df_a_coupled = df_a.loc[df_a['condition'] == 'Coupled']
    df_clean_a_coupled = df_clean_a.loc[df_clean_a['condition'] == 'Coupled']
    df_a_coupled_5 = df_a_coupled.loc[df_a_coupled['epoch'] == 5]
    df_clean_a_coupled_5 = df_clean_a_coupled.loc[df_clean_a_coupled['epoch'] == 5]
    
    
    df_b_coupled = df_b.loc[df_b['condition'] == 'Coupled']
    df_clean_b_coupled = df_clean_b.loc[df_clean_b['condition'] == 'Coupled']
    df_b_coupled_5 = df_b_coupled.loc[df_b_coupled['epoch'] == 5]
    df_clean_b_coupled_5 = df_clean_b_coupled.loc[df_clean_b_coupled['epoch'] == 5]
    
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 0, 10)
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 10, 20)
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 20, 30)
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 30, 40)
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 40, 50)
    ICA_result(df_clean_a_coupled_5, df_a_coupled_5, 1, 6720, 50, 64)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 0, 10)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 10, 20)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 20, 30)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 30, 40)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 40, 50)
    ICA_result(df_clean_b_coupled_5, df_b_coupled_5, 1, 6720, 50, 64)
    
    return df_b_coupled



