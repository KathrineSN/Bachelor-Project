# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:10:48 2021

@author: kathr
"""

from con_functions import *

#%% Loading preprocessed data from pair
epochs_a = mne.read_epochs('epochs_a_long_3.fif')
epochs_b = mne.read_epochs('epochs_b_long_3.fif')

epochs_a_3s = mne.read_epochs('epochs_a_3sec_3.fif')
epochs_b_3s = mne.read_epochs('epochs_b_3sec_3.fif')

epochs_a_s = mne.read_epochs('epochs_a_short_3.fif')
epochs_b_s = mne.read_epochs('epochs_b_short_3.fif')


#%% Load complex signal from 1, 3 and 25 seconds

theta, alpha, beta, angle, complex_signal_1s, epo_a_cleaned, epo_a = ccorr(epochs_a_s, epochs_b_s, 'pair003', 'short', drop_list = [])
theta, alpha, beta, angle, complex_signal_3s, epo_a_cleaned_3s, epo_a = ccorr(epochs_a_3s, epochs_b_3s, 'pair003', '3sec', drop_list = [])
theta, alpha, beta, angle, complex_signal_25s, epo_a_cleaned, epo_a = ccorr(epochs_a, epochs_b, 'pair003', 'long', drop_list = [])

#%%
# Get number of epochs relating to first epoch for 1 s

a = epo_a_cleaned.events[:,2]
d = dict()

for k, v in groupby(a):
    d.setdefault(k, []).append(len(list(v)))
print(d)

#%%
# concatenating 1 sec epochs

one_sec = complex_signal_1s[0][0][12][1][0:256]
two_sec = complex_signal_1s[0][1][12][1][0:256]
three_sec = complex_signal_1s[0][2][12][1][0:256]
four_sec = complex_signal_1s[0][3][12][1][0:256]
five_sec = complex_signal_1s[0][4][12][1][0:256]
six_sec = complex_signal_1s[0][5][12][1][0:256]

complex_s1_seq = np.concatenate((one_sec,two_sec,three_sec,four_sec,five_sec,six_sec))   

# Get number of epochs relating to first epoch for 3 s
first_3s = complex_signal_3s[0][0][12][1][0:768]
second_3s = complex_signal_3s[0][1][12][1][0:768]

complex_s3_seq = np.concatenate((first_3s,second_3s))   


#%% Plot phase and amplitude estimates from one participant 

plt.figure(figsize = (10,7))
t = np.arange(0,6,6/1536)
plt.subplot(3,1,1)
plt.title('Alpha signal')
plt.plot(t, complex_s1_seq)
plt.plot(t, complex_s3_seq)
plt.plot(t, complex_signal_25s[0][0][12][1][0:1536])
plt.xlabel('time (s)')
plt.subplot(3,1,2)
plt.title('Amplitude')
plt.plot(t, abs(complex_s1_seq), label = '1 seconds')
plt.plot(t, abs(complex_s3_seq), label = '3 seconds')
plt.plot(t, abs(complex_signal_25s[0][0][12][1][0:1536]), label = '25 seconds')
plt.xlabel('time (s)')
plt.legend(loc='upper right')
plt.subplot(3,1,3)
plt.title('Phase')
plt.plot(t, np.angle(complex_s1_seq))
plt.plot(t, np.angle(complex_s3_seq))
plt.plot(t, np.angle(complex_signal_25s[0][0][12][1][0:1536]))
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()




