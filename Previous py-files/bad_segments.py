# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:29:49 2021

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

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

#%%Loading raw

raw = mne.io.read_raw_bdf('Data\\pair003_20200129_1530.bdf', preload=True)

print(raw.info)

f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")

f_rawc = f_raw.copy()

#%%Picking right channel names

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

#%% Finding events

events = mne.find_events(f_raw, initial_event = True)
#print('Number of events:', len(events))
#print('Unique event codes:', np.unique(events[:, 2]))

new_events = events[1::2,:]
# Number of events

event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 } 


#%% Creating new event list
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
#%% Epoching

epochs_a_segmented = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                    baseline=(None, None), picks = picks_a, preload=True, detrend = None)

epochs_b_segmented = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 1,
                    baseline=(None, None), picks = picks_b, preload=True, detrend = None)

epochs_a_segmented.plot(n_epochs = 25, n_channels = 10)


#%% # Downsampling to 256 Hz 

epochs_a_seg_re = epochs_a_segmented.copy().resample(256, npad = 'auto')
epochs_b_seg_re = epochs_b_segmented.copy().resample(256, npad = 'auto')

epochs_a_seg_re.plot(n_epochs = 25, n_channels = 10)
epochs_b_seg_re.plot(n_epochs = 25, n_channels = 10)

#%% # Setting correct channel names

montage = mne.channels.make_standard_montage("biosemi64")
#montage.plot()
new_ch_names = montage.ch_names

for i in range(len(new_ch_names)):
    print(picks_a[i])
    print(new_ch_names[i])
    epochs_a_seg_re.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_a_seg_re.info.ch_names)


for i in range(len(new_ch_names)):
    
    epochs_b_seg_re.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})

print(epochs_b_seg_re.info.ch_names)

#%% Setting montage
epochs_a_seg_re.set_montage('biosemi64')
epochs_b_seg_re.set_montage('biosemi64')

#%% Setting bad channel

epochs_b_seg_re.info['bads'].append('FC1')

#%% ICA

ica1 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)
ica2 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)

ica1.fit(epochs_a_seg_re)
ica2.fit(epochs_b_seg_re)

ica1.plot_components()
ica2.plot_components()

ica1.plot_sources(epochs_a_seg_re, show_scrollbars=True)
ica2.plot_sources(epochs_b_seg_re, show_scrollbars=True)

epochs_a_seg_re_cleaned = ica1.apply(epochs_a_seg_re, exclude = [0,3])
epochs_b_seg_re_cleaned = ica2.apply(epochs_b_seg_re, exclude = [0,1,4,6])

epochs_b_seg_re_cleaned.interpolate_bads()

#%% Autoreject

cleaned_epochs_ICA = [epochs_a_seg_re_cleaned,epochs_b_seg_re_cleaned]
cleaned_epochs_AR, dic_AR = prep.AR_local(cleaned_epochs_ICA,
                                          strategy="union",
                                          threshold=50.0,
                                          verbose=True)

#%% Setting average reference
cleaned_epochs_AR[0].set_eeg_reference('average')
cleaned_epochs_AR[1].set_eeg_reference('average')

cleaned_epochs_AR[0].save('epochs_a_short.fif', overwrite = True)
cleaned_epochs_AR[1].save('epochs_b_short.fif', overwrite = True)

#%% Get timings of bad segments

Epoch_no = [381, 382, 554, 555, 557, 586, 587, 663, 678, 683, 967, 1126, 1146, 1303, 1351, 1378, 1408, 1439, 1661, 1662, 1663]

ev1 = ev_list[ev_list[:,2] != 101]
ev2 = ev1[ev1[:,2] != 104]
ev3 = ev2[ev2[:,2] != 106]
ev4 = ev3[ev3[:,2] != 1]
ev5 = ev4[Epoch_no,:]


# Creating annotations dictionary

events_to_bad = {102: 'Bad', 103: 'Bad', 105:'Bad',
              107 :'Bad', 108 :'Bad'}

An = mne.annotations_from_events(ev5, 2048, events_to_bad) 

# Set annotation duration

An.duration = np.ones(len(Epoch_no))

# Save annotations
An.save('bad_segments_pair003-annot.fif')










