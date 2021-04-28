# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:22:27 2021

@author: kathr
"""
import mne
import numpy as np
from matplotlib import pyplot as plt

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