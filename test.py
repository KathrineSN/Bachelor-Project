# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:59:20 2021

@author: kathr
"""
import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from hypyp import prep 

#%%
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
file_s = 'pair003_20200129_1530.bdf'

raw = mne.io.read_raw_bdf('Data\\' + file_s, preload=True)
    
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

#%% Three sec. epochs
# Number of events

event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 } 

# Creating new event list
event_list = []       
#3sec
ev_list = np.zeros((9*len(new_events),3))
#event_list.append(new_events[1,:])
for i in range(len(new_events)):
    temp = np.reshape(np.tile(new_events[i,:],9),(-1,3))
    temp[0,0]-=1.5*2048
    temp[1:9,0] += np.arange(start=2*2048, stop=8*(3*2048), step=3*2048)
    ev_list[i*9:(i+1)*9] = temp
    event_list.append(temp)
    
ev_list = ev_list.astype(int)

'''
# 6 seconds
ev_list = np.zeros((5*len(new_events),3))
#event_list.append(new_events[1,:])
for i in range(len(new_events)):
    temp = np.reshape(np.tile(new_events[i,:],5),(-1,3))
    temp[0,0]-=4*2048
    temp[1:9,0] += np.arange(start=2*2048, stop=4*(6*2048), step=6*2048)
    ev_list[i*5:(i+1)*5] = temp
    event_list.append(temp)
    
ev_list = ev_list.astype(int)
'''
fig = mne.viz.plot_events(ev_list, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)
fig.subplots_adjust(right=0.7)

#%% Downsampling
epochs_a = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 3,
                        baseline=(None, None), picks = picks_a, preload=True, detrend = None)

epochs_b = mne.Epochs(f_raw, ev_list, event_id = event_dict, tmin= 0, tmax= 3,
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
#%%
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

#%% Channelstuff
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

#epochs_a.plot(n_channels = 10, n_epochs = no_epochs_to_show)
#epochs_b.plot(n_channels = 10, n_epochs = no_epochs_to_show)
epo1.plot_psd(fmin = 2, fmax = 40)
epo2.plot_psd(fmin = 2, fmax = 40)
#epo1.info['bads'].append('FT7')
epo2.info['bads'].append('FC1')
#epo2.info['bads'].append('PO3')


#%% ICA epo1
ica1 = mne.preprocessing.ICA(n_components=15,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)
    
ica1.fit(epo1)
   
ica1.plot_components()

ica1.plot_sources(epo1, show_scrollbars=True)

#%% ICA epo2
ica2 = mne.preprocessing.ICA(n_components=15,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)
    
ica2.fit(epo2)
   
ica2.plot_components()

ica2.plot_sources(epo2, show_scrollbars=True)

#%% Cleaning

epo_cleaned1 = ica1.apply(epo1, exclude = [0,2])
epo_cleaned2 = ica2.apply(epo2, exclude = [0,4])
epo_cleaned1.interpolate_bads()
epo_cleaned2.interpolate_bads()
#cleaned_epochs_AR, dic_AR = prep.AR_local([epo_cleaned1, epo_cleaned2],
#                                      strategy="union",
#                                      threshold=50.0,
#                                      verbose=True)
epo_cleaned1.set_eeg_reference('average')
epo_cleaned2.set_eeg_reference('average')

epo_cleaned1.save('epochs_a_6sec_3.fif', overwrite = True)
epo_cleaned2.save('epochs_b_6sec_3.fif', overwrite = True)

#%% Connectivity calculations
from con_functions import *

#%% ccorr
drop_list_10 = [342, 351, 352, 353, 534, 603, 624, 625, 626, 832, 988, 1014, 1131, 1144, 1196, 1222, 1228, 1456, 1612, 1613, 1614]
theta, alpha, beta, angle_s, complex_signal_s_ccorr = ccorr(epochs_a_s, epochs_b_s, 'pair0010', 'short', drop_list = drop_list_10)

#%% coh
theta, alpha, beta, amp_s, complex_signal_s_coh = coh(epochs_a_s, epochs_b_s, 'pair0010', 'short', drop_list = drop_list_10)

#%%
#Checking length
epochs_a_s = mne.read_epochs('epochs_a_short_3.fif')
epochs_b_s = mne.read_epochs('epochs_b_short_3.fif')

#%%
#Calculating connectivity
theta, alpha, beta, angle_s, complex_signal, epo_cleaned, epo = ccorr(epochs_a_s, epochs_b_s, 'pair003', 'short', drop_list = [])


#%%
t = np.arange(0,1,1/256)
#realsig = signal_a
#ampsig = abs(complex_signal)**2
#plt.subplot(3,1,1)
plt.title('Alpha signal')
#plt.figure(figsize = (10,5))
plt.plot(t, complex_signal[0][0][12][1][0:256])

#plt.plot(t, complex_signal[0][0][12][1][0:256], t, complex_signal[1][0][12][1][0:256])
#plt.subplot(3,1,2)
#plt.title('Amplitude')
plt.plot(t, (abs(complex_signal))[0][0][12][1][0:256])

#plt.plot(t, (abs(complex_signal))[0][0][12][1][0:256],t, (abs(complex_signal))[1][0][12][1][0:256])
#plt.subplot(3,1,3)
#plt.title('Angle')
plt.plot(t, np.angle(complex_signal)[0][0][12][1][0:256]*10**-6)
#plt.plot(t, np.angle(complex_signal)[0][0][12][1][0:256]*10**-6,t, np.angle(complex_signal)[1][0][12][1][0:256]*10**-6)