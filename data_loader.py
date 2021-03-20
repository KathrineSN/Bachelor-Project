# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:35:23 2021

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

path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)

# Loading the data

raw = mne.io.read_raw_bdf('Data\\pair004_20200130_0930.bdf', preload=True)

print(raw.info)

#mne.find_events(raw1)

#raw.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)

#Filtering

f_raw = raw.filter(l_freq=1, h_freq=40, picks="eeg")

#f_rawc = f_raw.copy()

#for i in range(len(new_ch_names)):
#    print(picks_a[i])
#    print(new_ch_names[i])
#    f_raw.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})
#
#print(f_raw.info.ch_names)

#ica1.plot_sources(f_raw.pick(new_ch_names), show_scrollbars=True)
#ica2.plot_sources(f_raw.pick(new_ch_names), show_scrollbars=True)

#icas[0].plot_sources(f_raw.pick(new_ch_names), show_scrollbars=True)
#icas[1].plot_sources(f_raw.pick(new_ch_names), show_scrollbars=True)

#icas[0].plot_overlay(f_raw.pick(new_ch_names), exclude = [0])
#icas[0].plot_overlay(f_raw.pick(new_ch_names), exclude = [0], picks = new_ch_names, start = 997527, stop = 1000000)
#icas[0].plot_overlay(f_raw.pick(new_ch_names), exclude = [0,3], picks = new_ch_names, start = 1802598, stop = 1853805)
#icas[0].plot_overlay(f_raw.pick(new_ch_names), exclude = [0,3])

#icas[1].plot_overlay(f_raw.pick(new_ch_names), exclude = [0,1,4,6], picks = new_ch_names, start = 941320, stop = 992527)

#icas[1].plot_sources(f_raw.pick(new_ch_names), show_scrollbars=True)

#f_raw.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)

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
#epo2.info['bads'].append('FC1')
#epo2_ex = epo2.copy()
#epo2_ex.
#epo2.interpolate_bads()

#ICA

ica1 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)
ica2 = mne.preprocessing.ICA(n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)

ica1.fit(epo1)
ica2.fit(epo2)

ica1.plot_components()
ica2.plot_components()

ica1.plot_sources(epo1, show_scrollbars=True)
ica2.plot_sources(epo2, show_scrollbars=True)

epo1_cleaned = ica1.apply(epo1, exclude = [0,3])
epo2_cleaned = ica2.apply(epo2, exclude = [0,8])
epo1_cleaned.save('epochs_a_cleaned004-epo.fif', overwrite = True)
epo2_cleaned.save('epochs_b_cleaned004-epo.fif', overwrite = True)

#%%
######
icas = prep.ICA_fit([epo1, epo2],
                    n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)


epo1_cleaned = icas[0].apply(epo1, exclude = [0,3])
epo2_cleaned = icas[1].apply(epo2, exclude = [0,1,4,6])

epo1_cleaned.save('epochs_a_cleaned-epo.fif', overwrite = True)
epo2_cleaned.save('epochs_b_cleaned-epo.fif', overwrite = True)
epo1_cleaned.to_data_frame()
epo2_cleaned.to_data_frame()

#%%
#####################################################

#To create raw with ICA applied
f_rawa = f_raw.pick(picks_a)
f_rawb = f_rawc.pick(picks_b)
f_rawa_c = f_rawa.copy()
f_rawb_c = f_rawb.copy()
                     
for i in range(len(new_ch_names)):
    print(picks_a[i])
    print(new_ch_names[i])
    f_rawa.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(f_rawa.info.ch_names)


for i in range(len(new_ch_names)):
    
    f_rawb.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})

print(f_rawb.info.ch_names)

for i in range(len(new_ch_names)):
    print(picks_a[i])
    print(new_ch_names[i])
    f_rawa_c.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(f_rawa_c.info.ch_names)


for i in range(len(new_ch_names)):
    
    f_rawb_c.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})

print(f_rawb_c.info.ch_names)

rawa_cleaned = icas[0].apply(f_rawa_c, exclude = [0,3])
rawb_cleaned = icas[1].apply(f_rawb_c, exclude = [0,1,4,6])


cleaned_epochs_ICA = [epo1_cleaned,epo2_cleaned]

#cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])



cleaned_epochs_AR, dic_AR = prep.AR_local(cleaned_epochs_ICA,
                                          strategy="union",
                                          threshold=50.0,
                                          verbose=True)

# Plot the two sources

icas[0].plot_sources(epo1, show_scrollbars=True)
icas[1].plot_sources(epo2, show_scrollbars=True)

# Interactive plots
icas[0].plot_components(inst = epo1)
icas[1].plot_components(inst = epo2)

########### Plotting raw before and after ICA ##########
df_beforeICA_a = f_rawa.to_data_frame()
df_beforeICA_b = f_rawb.to_data_frame()

df_afterICA_a = rawa_cleaned.to_data_frame()
df_afterICA_b = rawb_cleaned.to_data_frame()

# Plot cleaned epochs
cleaned_epochs_AR[0].plot(n_epochs = 1, n_channels = 10) 
cleaned_epochs_AR[1].plot(n_epochs = 1, n_channels = 10) 


df_a_clean = cleaned_epochs_AR[0].to_data_frame()
df_b_clean = cleaned_epochs_AR[1].to_data_frame()

df_a_clean[(df_a_clean == 'Coupled').any(axis=1)].plot(x = 'time', y = new_ch_names)

df_a_clean_c = df_a_clean[(df_a_clean == 'Coupled').any(axis=1)]                                 
df_a_c = df_a[(df_a == 'Coupled').any(axis=1)]                                 

#df_beforeICA_a.to_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\before_ica_a.pickle")
#df_afterICA_a.to_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\after_ica_a.pickle")
#df_beforeICA_b.to_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\before_ica_b.pickle")
#df_afterICA_b.to_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\after_ica_b.pickle")

df_beforeICA_a = pd.read_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\before_ica_a.pickle")
df_afterICA_a = pd.read_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\after_ica_a.pickle")
df_beforeICA_b = pd.read_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\before_ica_b.pickle")
df_afterICA_b = pd.read_pickle("C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt\\after_ica_b.pickle")

#Plotting loop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

def ICA_result(df_afterICA, df_beforeICA, start, length, chan_start, chan_end):
    signals_after = []
    signals_before = []
    for i in range(len(new_ch_names)):
        
        df_afterICA = df_afterICA.iloc[start:]
        df_beforeICA = df_beforeICA.iloc[start:]

        #t = df_afterICA.loc[(df_afterICA['time'] >= start) & (df_afterICA['time'] <= stop)]
        t = (df_afterICA[['time']].head(length))
        #signals_after.append(df_afterICA.loc[(df_afterICA[new_ch_names[i]] >= start) & (df_afterICA[new_ch_names[i]] <= stop)])
        #signals_before.append(df_beforeICA.loc[(df_beforeICA[new_ch_names[i]] >= start) & (df_beforeICA[new_ch_names[i]] <= stop)])

        signals_after.append(df_afterICA[[new_ch_names[i]]].head(length))
        signals_before.append(df_beforeICA[[new_ch_names[i]]].head(length))

    #print(t)
    #print(t[len(t)-1])
    
    fig = plt.figure()
    no_chan = chan_end-chan_start
    for i in range(no_chan):
        temp = i
        ax = plt.subplot(no_chan,1,i+1)
        plt.plot(t,signals_before[chan_start + i], t, signals_after[chan_start + i])
        plt.subplots_adjust(hspace = .001)
        temp = tic.MaxNLocator(3)
        ax.yaxis.set_major_locator(temp)
        ax.set_xticklabels(())
        ax.title.set_visible(False)
        ax.set_ylabel(new_ch_names[chan_start + i])
    
    #fig.set_xlabel('time')
    return plt.show()

t = df_afterICA_b[['time']]
print(t[:1000])
print(t[:100000])

ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 0, 10)
ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 10, 20)
ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 20, 30)
ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 30, 40)
ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 40, 50)
ICA_result(df_afterICA_b, df_beforeICA_b, 1000, 100000, 50, 64)

ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 0, 10)
ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 10, 20)
ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 20, 30)
ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 30, 40)
ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 40, 50)
ICA_result(df_afterICA_a, df_beforeICA_a, 1000, 100000, 50, 64)


####################
                                 
fig = plt.figure()
t = df_afterICA_a[['time']].head(100000)
s1 = df_afterICA_a[['Fp1']].head(100000)
s2 = df_afterICA_a[['AF7']].head(100000)
s3 = df_afterICA_a[['AF3']].head(100000)
s4 = df_afterICA_a[['F1']].head(100000)

s11 = df_beforeICA_a[['Fp1']].head(100000)
s22 = df_beforeICA_a[['AF7']].head(100000)
s33 = df_beforeICA_a[['AF3']].head(100000)
s44 = df_beforeICA_a[['F1']].head(100000)



yprops = dict(rotation=0,
              horizontalalignment='right',
              verticalalignment='center',
              x=-0.01)

axprops = dict(yticks=[])

ax1 =fig.add_axes([0.1, 0.7, 0.8, 0.2], **axprops)
ax1.plot(t, s11, t, s1)
ax1.set_ylabel('Fp1', **yprops)

axprops['sharex'] = ax1
axprops['sharey'] = ax1
# force x axes to remain in register, even with toolbar navigation
ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.2], **axprops)

ax2.plot(t, s22, t, s2)
ax2.set_ylabel('AF7', **yprops)

ax3 = fig.add_axes([0.1, 0.3, 0.8, 0.2], **axprops)
ax3.plot(t, s33, t, s3)
ax3.set_ylabel('AF3', **yprops)

ax4 = fig.add_axes([0.1, 0.1, 0.8, 0.2], **axprops)
ax4.plot(t, s44, t, s4)
ax4.set_ylabel('F1', **yprops)

# turn off x ticklabels for all but the lower axes
for ax in ax1, ax2, ax3:
    plt.setp(ax.get_xticklabels(), visible=False)

plt.show()
                                 
# Setting the average reference

cleaned_epochs_AR[0].set_eeg_reference('average')
cleaned_epochs_AR[1].set_eeg_reference('average')

# Saving cleaned epochs
#cleaned_epochs_AR[0].save('epochs_cleaned_pair003_a-epo.fif', overwrite = True)
#cleaned_epochs_AR[1].save('epochs_cleaned_pair003_b-epo.fif', overwrite = True)


'''
################# Remove from here
from collections import OrderedDict
#Defining frequency bands
freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}

freq_bands = OrderedDict(freq_bands)

sampling_rate = epo1.info['sfreq']

#Picking preproccessed epochs for each participant
preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]

#Power spectral density
psd1 = analyses.pow(preproc_S1, fmin=7.5, fmax=11,
                    n_fft=1000, n_per_seg=1000, epochs_average=True)
psd2 = analyses.pow(preproc_S2, fmin=7.5, fmax=11,
                    n_fft=1000, n_per_seg=1000, epochs_average=True)
data_psd = np.array([psd1.psd, psd2.psd])

#Connectivity

#Data and storage
data_inter = np.array([preproc_S1, preproc_S2])
result_intra = []

#Analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                             freq_bands)



result = analyses.compute_sync(complex_signal, mode='ccorr')


#Get interbrain part of the matrix
n_ch = len(epo1.info['ch_names'])
theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch:2*n_ch]

# Alpha low for example
values = alpha_low
values -= np.diag(np.diag(values))

C = (values - np.mean(values[:])) / np.std(values[:])

#Slicing results to get the intra-brain part of matrix
for i in [0, 1]:
    theta, alpha_low, alpha_high, beta, gamma = result[:, i:i+n_ch, i:i+n_ch]
    # choosing Alpha_Low for futher analyses for example
    values_intra = alpha_low
    values_intra -= np.diag(np.diag(values_intra))
    # computing Cohens'D for further analyses for example
    C_intra = (values_intra -
               np.mean(values_intra[:])) / np.std(values_intra[:])
    # can also sample CSD values directly for statistical analyses
    result_intra.append(C_intra)


### Comparing inter-brain connectivity values to random signal
data = [np.array([values, values]), np.array([result_intra[0], result_intra[0]])]

statscondCluster = stats.statscondCluster(data=data,
                                          freqs_mean=np.arange(7.5, 11),
                                          ch_con_freq=None,
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)

#Visualization of interbrain connectivity in 2D
viz.viz_2D_topomap_inter(epo1, epo2, C, threshold='auto', steps=10, lab=True)

#Visualization of interbrain connectivity in 3D
viz.viz_3D_inter(epo1, epo2, C, threshold='auto', steps=10, lab=False)
'''








