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

path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project\\Files from preprocessing"
os.chdir(path)

epochs_a = mne.read_epochs('epochs_a_long_004.fif', preload = True)
epochs_b = mne.read_epochs('epochs_b_long_004.fif', preload = True)

epochs_a.plot_psd(fmin = 2, fmax = 40, average = True)
epochs_b.plot_psd(fmin = 2, fmax = 40, average = True)

# Frequency plot for each condition

f, ax = plt.subplots()
psds1, freqs1 = mne.time_frequency.psd_multitaper(epochs_a['Coupled'], fmin=2, fmax=40)
psds1 = 10. * np.log10(psds1)
psds_mean1 = psds1.mean(0).mean(0)

psds2, freqs2 = mne.time_frequency.psd_multitaper(epochs_a['Uncoupled'], fmin=2, fmax=40)
psds2 = 10. * np.log10(psds2)
psds_mean2 = psds2.mean(0).mean(0)

psds3, freqs3 = mne.time_frequency.psd_multitaper(epochs_a['Leader'], fmin=2, fmax=40)
psds3 = 10. * np.log10(psds3)
psds_mean3 = psds3.mean(0).mean(0)

psds4, freqs4 = mne.time_frequency.psd_multitaper(epochs_a['Follower'], fmin=2, fmax=40)
psds4 = 10. * np.log10(psds4)
psds_mean4 = psds4.mean(0).mean(0)

psds5, freqs5 = mne.time_frequency.psd_multitaper(epochs_a['Control'], fmin=2, fmax=40)
psds5 = 10. * np.log10(psds5)
psds_mean5 = psds5.mean(0).mean(0)

ax.plot(freqs1, psds_mean1, label = "Coupled")
ax.plot(freqs2, psds_mean2, label = "Uncoupled")
ax.plot(freqs3, psds_mean3, label = "Leader")
ax.plot(freqs4, psds_mean4, label = "Follower")
ax.plot(freqs5, psds_mean5, label = "Control") 
ax.legend()
ax.set(title='Multitaper PSD', xlabel='Frequency (Hz)',
       ylabel='Power Spectral Density (dB)')
plt.show()


f, ax = plt.subplots()
psds1, freqs1 = mne.time_frequency.psd_multitaper(epochs_b['Coupled'], fmin=2, fmax=40)
psds1 = 10. * np.log10(psds1)
psds_mean1 = psds1.mean(0).mean(0)

psds2, freqs2 = mne.time_frequency.psd_multitaper(epochs_b['Uncoupled'], fmin=2, fmax=40)
psds2 = 10. * np.log10(psds2)
psds_mean2 = psds2.mean(0).mean(0)

psds3, freqs3 = mne.time_frequency.psd_multitaper(epochs_b['Leader'], fmin=2, fmax=40)
psds3 = 10. * np.log10(psds3)
psds_mean3 = psds3.mean(0).mean(0)

psds4, freqs4 = mne.time_frequency.psd_multitaper(epochs_b['Follower'], fmin=2, fmax=40)
psds4 = 10. * np.log10(psds4)
psds_mean4 = psds4.mean(0).mean(0)

psds5, freqs5 = mne.time_frequency.psd_multitaper(epochs_b['Control'], fmin=2, fmax=40)
psds5 = 10. * np.log10(psds5)
psds_mean5 = psds5.mean(0).mean(0)

ax.plot(freqs1, psds_mean1, label = "Coupled")
ax.plot(freqs2, psds_mean2, label = "Uncoupled")
ax.plot(freqs3, psds_mean3, label = "Leader")
ax.plot(freqs4, psds_mean4, label = "Follower")
ax.plot(freqs5, psds_mean5, label = "Control") 
ax.legend()
ax.set(title='Multitaper PSD', xlabel='Frequency (Hz)',
       ylabel='Power Spectral Density (dB)')
plt.show()


