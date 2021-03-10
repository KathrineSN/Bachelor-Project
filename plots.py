# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:02:37 2021

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
import matplotlib.ticker as tic

path="C:\\Users\\kathr\\OneDrive\\Documents\\Bachelor projekt"
os.chdir(path)

# Loading data

epochs_a_resampled = mne.read_epochs('epochs_a_resampled-epo.fif', preload = True)
epochs_b_resampled = mne.read_epochs('epochs_b_resampled-epo.fif', preload = True)

epochs_a_cleaned = mne.read_epochs('epochs_a_cleaned-epo.fif', preload = True)
epochs_b_cleaned = mne.read_epochs('epochs_b_cleaned-epo.fif', preload = True)




plt.Figure()
epochs_a_resampled.plot(n_epochs = 1, n_channels = 10)
epochs_a_cleaned.plot(n_epochs = 1, n_channels = 10)
plt.show()

# Creating dfs

df_a = epochs_a_resampled.to_data_frame()
df_b = epochs_b_resampled.to_data_frame()

df_clean_a = epochs_a_cleaned.to_data_frame()
df_clean_b = epochs_b_cleaned.to_data_frame()

montage = mne.channels.make_standard_montage("biosemi64")
#montage.plot()
new_ch_names = montage.ch_names

df_a_coupled = df_a.loc[df_a['condition'] == 'Coupled']
df_clean_a_coupled = df_clean_a.loc[df_clean_a['condition'] == 'Coupled']
df_a_coupled_5 = df_a_coupled.loc[df_a_coupled['epoch'] == 5]
df_clean_a_coupled_5 = df_clean_a_coupled.loc[df_clean_a_coupled['epoch'] == 5]


df_b_coupled = df_b.loc[df_b['condition'] == 'Coupled']
df_clean_b_coupled = df_clean_b.loc[df_clean_b['condition'] == 'Coupled']
df_b_coupled_5 = df_b_coupled.loc[df_b_coupled['epoch'] == 5]
df_clean_b_coupled_5 = df_clean_b_coupled.loc[df_clean_b_coupled['epoch'] == 5]


fig = plt.figure()
ax1 = fig.add_subplot(111)
t = df_clean_a_coupled_5[['time']]
s1 = df_clean_a_coupled_5[['Fp1']]
s2 = df_a_coupled_5[['Fp1']]

ax1.plot(t, s1, t, s2)
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('Fp1')




def ICA_result(df_afterICA, df_beforeICA, start, length, chan_start, chan_end):
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
        temp = i
        ax = plt.subplot(no_chan,1,i+1)
        plt.plot(t ,signals_before[chan_start + i], t, signals_after[chan_start + i])
        plt.subplots_adjust(hspace = .001)
        #temp = tic.MaxNLocator(3)
        #ax.yaxis.set_major_locator(temp)
        #ax.set_xticklabels(('time'))
        ax.title.set_visible(False)
        ax.set_ylabel(new_ch_names[chan_start + i])
        ax.set_xlabel('time (ms)')
        
    
    #plt.set_xlabel('time')
    return plt.show()

#t = df_afterICA_b[['time']]
#print(t[:1000])
#print(t[:100000])

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

epochs_b_cleaned.interpolate_bads()

epochs_a_cleaned.set_eeg_reference('average')
epochs_b_cleaned.set_eeg_reference('average')

epochs_a_cleaned.plot(n_channels = 10, n_epochs = 1)
epochs_b_cleaned.plot(n_channels = 10, n_epochs = 1)

#epochs_a_cleaned.save('epochs_a_preprocessed-epo.fif', overwrite = True)
#epochs_b_cleaned.save('epochs_b_preprocessed-epo.fif', overwrite = True)




