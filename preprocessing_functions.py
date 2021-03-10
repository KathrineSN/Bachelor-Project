# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:35:53 2021

@author: kathr
"""

#Importing dependencies
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

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

#%% Loading raw data, filter

def loading_raw(file):
    
    #Loading raw file
    raw = mne.io.read_raw_bdf('Data\\'+ file, preload=True)
    
    #High pass & low pass filter
    f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")

    return f_raw



#%% Creating long epochs

def long_epochs(raw, plot = True):
    
    #Changing channel names
    picks_a = []
    picks_b = []
    channels = raw.info.ch_names
    
    for i in range(len(channels)):
        if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
            picks_a.append(channels[i])
    
    for i in range(len(channels)):
        if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
            picks_b.append(channels[i])
            
    # Finding event & picking out beginning of trials
    events = mne.find_events(f_raw, initial_event = True)
    print('Number of events:', len(events))
    print('Unique event codes:', np.unique(events[:, 2]))

    new_events = events[1::2,:]
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 } 

    epochs_a = mne.Epochs(f_raw, new_events, event_id = event_dict, tmin=-1.5, tmax=25,
                    baseline=(None, 0), picks = picks_a, preload=True, detrend = None)

    

    epochs_b = mne.Epochs(f_raw, new_events, event_id = event_dict, tmin=-1.5, tmax=25,
                    baseline=(None, 0), picks = picks_b, preload=True, detrend = None)

    if plot:
        epochs_a.plot(n_epochs = 1, n_channels = 10)
        epochs_b.plot(n_epochs = 1, n_channels = 10)
    
    #Setting correct channel names
    montage = mne.channels.make_standard_montage("biosemi64")
    
    new_ch_names = montage.ch_names
    
    for i in range(len(new_ch_names)):
        epochs_a.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})
        
    
    for i in range(len(new_ch_names)):    
        epochs_b.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
        
    #Setting montage 
    epochs_a.set_montage('biosemi64')
    epochs_b.set_montage('biosemi64')
    
    return events, epochs_a, epochs_b


#%% Creating short epochs

def short_epochs(raw, plot = True):
    
    #Changing channel names
    picks_a = []
    picks_b = []
    channels = raw.info.ch_names
    
    for i in range(len(channels)):
        if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
            picks_a.append(channels[i])
    
    for i in range(len(channels)):
        if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
            picks_b.append(channels[i])
    
    #finding events
    events = mne.find_events(raw, initial_event = True)

    new_events = events[1::2,:]
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
                  'Follower': 107, 'Control':108} 
    
    #Creating new event list for shorter epochs
    event_list = []
    ev_list = np.zeros((26*len(new_events),3))
    
    for i in range(len(new_events)):
        temp = np.reshape(np.tile(new_events[i,:],26),(-1,3))
        temp[0,0]-=1.5*2048
        temp[1:26,0] += np.arange(start=0, stop=25*2048, step=2048)
        ev_list[i*26:(i+1)*26] = temp
        event_list.append(temp)
        
    ev_list = ev_list.astype(int)
    
    epochs_a = mne.Epochs(raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                    baseline=(None, None), picks = picks_a, preload=True, detrend = None)

    epochs_b = mne.Epochs(raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                        baseline=(None, None), picks = picks_b, preload=True, detrend = None)
    
    
    if plot:
        fig = mne.viz.plot_events(ev_list, sfreq=raw.info['sfreq'],
                               first_samp=raw.first_samp)
        fig.subplots_adjust(right=0.7)
        epochs_a.plot(n_epochs = 25, n_channels = 10)
        epochs_b.plot(n_epochs = 25, n_channels = 10)
        
    #Setting correct channel names
    montage = mne.channels.make_standard_montage("biosemi64")
    
    new_ch_names = montage.ch_names
    
    for i in range(len(new_ch_names)):

        epochs_a.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})
         
    for i in range(len(new_ch_names)):
        
        epochs_b.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
        
    #Setting montage 
    epochs_a.set_montage('biosemi64')
    epochs_b.set_montage('biosemi64')
    
    return ev_list, epochs_a, epochs_b


#%% Downsampling 

def downsampling(epochs_a, epochs_b):
    
    epochs_a_resampled = epochs_a.copy().resample(256, npad = 'auto')
    epochs_b_resampled = epochs_b.copy().resample(256, npad = 'auto')
    
    return epochs_a_resampled, epochs_b_resampled


#%% Bad channels and channel statistics

def find_bad_chans(epochs_a, epochs_b, no_epochs_to_show, plot = True):
    
    montage = mne.channels.make_standard_montage("biosemi64")  
    new_ch_names = montage.ch_names
    
    df_a = epochs_a.to_data_frame(picks = new_ch_names)
    df_b = epochs_b.to_data_frame(picks = new_ch_names)
    ch_stat_a = df_a.describe()
    ch_stat_b = df_b.describe()    
    
    if plot:
        epochs_a.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        epochs_b.plot(n_channels = 10, n_epochs = no_epochs_to_show)
        epochs_a.plot_psd(fmin = 2, fmax = 40)
        epochs_b.plot_psd(fmin = 2, fmax = 40)
    
    return ch_stat_a, ch_stat_b


#%% 

def calculate_ICA(epochs_a, epochs_b, plot = True):
    
    epochs_a.set_montage('biosemi64')
    epochs_b.set_montage('biosemi64')
    
    ica1 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)
    
    ica2 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)
    
    ica1.fit(epochs_a)
    ica2.fit(epochs_b)

    if plot: 

        ica1.plot_components()
        ica2.plot_components()

        ica1.plot_sources(epochs_a, show_scrollbars=True)
        ica2.plot_sources(epochs_b, show_scrollbars=True)
        
    return ica1, ica2

#%% 

def ICA_remove_components(epochs_a, epochs_b, ica1, ica2, ic_remove1, ic_remove2):
    epochs_a_new = epochs_a.copy()
    epochs_b_new = epochs_a.copy()
    epochs_a_cleaned = ica1.apply(epochs_a_new, exclude = ic_remove1)
    epochs_b_cleaned = ica2.apply(epochs_b_new, exclude = ic_remove2)
    
    return epochs_a_cleaned, epochs_b_cleaned
    

#%%

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

    #print(t)
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
    
    return 

#%% Autoreject for short epochs

def remove_bad_segments(epochs_a, epochs_b):
    
    # This function is to be used after ICA only
    cleaned_epochs_AR, dic_AR = prep.AR_local([epochs_a, epochs_b],
                                              strategy="union",
                                              threshold=50.0,
                                              verbose=True)
    
    return cleaned_epochs_AR[0], cleaned_epochs_AR[1], dic_AR

#%% Re-referencing 

def set_reference(epochs_a, epochs_b):
    epochs_a.set_eeg_reference('average')
    epochs_b.set_eeg_reference('average')
    
    return epochs_a, epochs_b
    
#%% Interpolate Bads

def interpolate_bad_chans(epochs_a, epochs_b):
    
    if epochs_a.info['bads'] != []:
        epochs_a.interpolate_bads()
        print('Interpolated bad channel(s) of participant a')
        
    if epochs_b.info['bads'] != []:
        epochs_b.interpolate_bads()
        print('Interpolated bad channel(s) of participant b')
        

#%% Pipeline part 1

# Loading raw
f_raw = loading_raw('pair0010_20200205_1230.bdf')

# Creating long epochs
events, epochs_a, epochs_b = long_epochs(f_raw)

# Creating short epochs
ev_list, epochs_a_s, epochs_b_s = short_epochs(f_raw)

# Downsampling
epochs_a_resampled, epochs_b_resampled = downsampling(epochs_a, epochs_b)
epochs_a_s_resampled, epochs_b_s_resampled = downsampling(epochs_a_s, epochs_b_s)

# Identifying bad channels
ch_stat_a, ch_stat_b = find_bad_chans(epochs_a_resampled, epochs_b_resampled, 1)
ch_stat_a_s, ch_stat_b_s = find_bad_chans(epochs_a_s_resampled, epochs_b_s_resampled,10)

#%% Pipeline part 2

# Marking bad channels
#epochs_a_resampled.info['bads'].append('PO3')
#epochs_a_s_resampled.info['bads'].append('PO3')


# Calculating IC's
ica1, ica2 = calculate_ICA(epochs_a_resampled, epochs_b_resampled, plot = True)
ica1_s, ica2_s = calculate_ICA(epochs_a_s_resampled, epochs_b_s_resampled, plot = True)


#%% Pipeline part 3
# Removing IC's related to artifacts
epochs_a_cleaned, epochs_b_cleaned = ICA_remove_components(epochs_a_resampled, epochs_b_resampled, ica1, ica2)
epochs_a_s_cleaned, epochs_b_s_cleaned = ICA_remove_components(epochs_a_s_resampled, epochs_b_s_resampled, ica1_s, ica2_s)

# Plot showing effect of ICA
before_vs_after_ICA(epochs_a_resampled, epochs_b_resampled, epochs_a_cleaned, epochs_b_cleaned)

# Autorejection only on short epochs
epochs_a_s_cleaned, epochs_b_s_cleaned, dic_AR = remove_bad_segments(epochs_a_s_cleaned, epochs_b_s_cleaned)

#Setting the average reference
epochs_a_cleaned, epochs_b_cleaned = set_reference(epochs_a_cleaned, epochs_b_cleaned)
epochs_a_s_cleaned, epochs_b_s_cleaned = set_reference(epochs_a_s_cleaned, epochs_b_s_cleaned)

#Saving the epochs
epochs_a_cleaned.save('epochs_a_long_004.fif', overwrite = True)
epochs_b_cleaned.save('epochs_b_long_004.fif', overwrite = True)

epochs_a_s_cleaned.save('epochs_a_short_004.fif', overwrite = True)
epochs_b_s_cleaned.save('epochs_b_short_004.fif', overwrite = True)


    
    
    
    
            





