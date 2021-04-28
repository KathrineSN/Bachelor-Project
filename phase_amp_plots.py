# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:57:37 2021

@author: kathr
"""

import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Bachelor-Project"
os.chdir(path)
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
from con_functions import *

#%% Loading a pair from long

epochs_a = mne.read_epochs('epochs_a_long_10_resting.fif')
epochs_b = mne.read_epochs('epochs_b_long_10_resting.fif')


#%% ccorr

theta, alpha, beta, angle, complex_signal = ccorr(epochs_a, epochs_b, 'pair0010', 'long', drop_list = [])

#%% coh
theta, alpha, beta, amp = coh(epochs_a, epochs_b, 'pair0010', 'long', drop_list = [])

#%% PLV/phase

freq_bands = {'Theta': [4, 7],'Alpha' :[8, 13],'Beta': [15, 25]}

freq_bands = OrderedDict(freq_bands)

sampling_rate = epochs_a.info['sfreq']

#Connectivity

#Data and storage
data_inter = np.array([epochs_a['Control'], epochs_b['Control']])

#Analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                             freq_bands)

result, angle, _, phase = analyses.compute_sync(complex_signal, mode='plv', epochs_average = True)


#%% Loading a pair from short
epochs_a_s = mne.read_epochs('epochs_a_short_10_resting.fif')
epochs_b_s = mne.read_epochs('epochs_b_short_10_resting.fif')

#%% ccorr
drop_list_10 = [342, 351, 352, 353, 534, 603, 624, 625, 626, 832, 988, 1014, 1131, 1144, 1196, 1222, 1228, 1456, 1612, 1613, 1614]
theta, alpha, beta, angle_s, complex_signal_s = ccorr(epochs_a_s, epochs_b_s, 'pair0010', 'short', drop_list = drop_list_10)

#%% coh
theta, alpha, beta, amp_s, complex_signal_s_coh = coh(epochs_a_s, epochs_b_s, 'pair0010', 'short', drop_list = drop_list_10)

#%% Extracting angle from participant a and b respectively

#long
signal_a = complex_signal[0][0][12][1][0:256]
signal_b = complex_signal[1][0][12][1][0:256]

angle_a = angle[0][1][12][0:256]
angle_b = angle[0][1][75][0:256]

amp_a = amp[0][1][12][0:256]
amp_b = amp[0][1][75][0:256]

phase_a = phase[0][1][12][0:256]
phase_b = phase[0][1][75][0:256]

#short
signal_a_s = complex_signal_s[0][0][12][1][:]
signal_b_s = complex_signal_s[1][0][12][1][:]

angle_a_s = []
angle_b_s = []
for i in range(23):
    angle_a_s.append(angle_s[i][1][12][:]) 
    angle_b_s.append(angle_s[i][1][75][:])
angle_a_s = np.concatenate(angle_a_s)[0:256]
angle_b_s = np.concatenate(angle_b_s)[0:256]

amp_a_s = []
amp_b_s = []
for i in range(23):
    amp_a_s.append(amp_s[i][1][12][:]) 
    amp_b_s.append(amp_s[i][1][75][:])
amp_a_s = np.concatenate(amp_a_s)[0:256]
amp_b_s = np.concatenate(amp_b_s)[0:256]


#%% Extracting amplitude from each participant

#t = range(1000,2000)
t = np.arange(0,256,1)

plt.subplot(6,1,1)
plt.title('alpha signal long')
plt.plot(t, np.real(signal_a), t, np.real(signal_b))
plt.subplot(6,1,2)
plt.title('amplitude long')
plt.plot(t, amp_a, t, amp_b)
plt.subplot(6,1,3)
plt.title('angle long')
plt.plot(t, angle_a, t, angle_b)
plt.subplot(6,1,4)
plt.title('alpha signal short')
plt.plot(t, np.real(signal_a_s), t, np.real(signal_b_s))
plt.subplot(6,1,5)
plt.title('amplitude short')
plt.plot(t, amp_a_s, t, amp_b_s)
plt.subplot(6,1,6)
plt.title('angle short')
plt.plot(t, angle_a_s, t, angle_b_s)

#%% Plot for methods section

t = np.arange(0,1,1/256)
plt.subplot(3,1,1)
plt.title('Alpha signal')
plt.plot(t, np.real(signal_a))
plt.plot(t, np.real(signal_b))
plt.xlabel('time (s)')
plt.subplot(3,1,2)
plt.title('Amplitude')
plt.plot(t, amp_a, label = 'Participant a')
plt.plot(t,amp_b, label = 'Participant b')
plt.xlabel('time (s)')
plt.legend(loc='upper right')
plt.subplot(3,1,3)
plt.title('Phase')
plt.plot(t, angle_a, t, angle_b)
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()

#%% Plot checking that amplitude is right - FIX THIS
t = np.arange(0,1,1/256)

#realsig = signal_a
#ampsig = abs(complex_signal)**2
plt.subplot(3,1,1)
#plt.figure(figsize = (10,5))
plt.plot(t, complex_signal[0][0][12][1][0:256])
plt.subplot(3,1,2)
plt.plot(t, (abs(complex_signal)**2)[0][0][12][1][0:256])
plt.subplot(3,1,3)
plt.plot(t, np.angle(complex_signal)[0][0][12][1][0:256])

#%%



