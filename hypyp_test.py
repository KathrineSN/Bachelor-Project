# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:56:52 2021

@author: kathr
"""
#Core
import io
from copy import copy
from collections import OrderedDict
import requests

#Data science
import numpy as np
import scipy

#Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.mesh import Mesh
from hypyp.ext.mpl3d.camera import Camera

#MNE
import mne

#HyPyP
from hypyp import prep # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import stats
from hypyp import viz

#Defining frequency bands
freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}

freq_bands = OrderedDict(freq_bands)

#Loading in the data
URL_TEMPLATE = "https://github.com/ppsp-team/HyPyP/blob/master/data/participant{}-epo.fif?raw=true"

def get_data(idx):
    return io.BytesIO(requests.get(URL_TEMPLATE.format(idx)).content)

epo1 = mne.read_epochs(
    get_data(1),
    preload=True,
) 

epo2 = mne.read_epochs(
    get_data(2),
    preload=True,
)

#Ensure equal number of epochs between two participants
mne.epochs.equalize_epoch_counts([epo1, epo2])

#Specify sampling frequency
sampling_rate = epo1.info['sfreq'] #Hz

## Preprocessing of epochs

#ICA
icas = prep.ICA_fit([epo1, epo2],
                    n_components=15,
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42)


cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])



cleaned_epochs_AR, dic_AR = prep.AR_local(cleaned_epochs_ICA,
                                          strategy="union",
                                          threshold=50.0,
                                          verbose=True)

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



result = analyses.compute_sync(complex_signal, mode='coh')


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






