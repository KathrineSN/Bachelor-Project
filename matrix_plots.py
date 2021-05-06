# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:04:07 2021

@author: kathr
"""
from con_functions import load_avg_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
#%% Heatmap definition
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #cbar.ax.rc('axes', labelsize=11)    # fontsize of the x and y labels
    cbar.ax.tick_params(labelsize=11)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
#%% Setting fintsizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%%  Coherence Connectivity for Coupled (25 sec. epochs)
# Plot it out 
coupled = load_avg_matrix('coh','beta', 'Coupled', 'long', save = 1) 
fig, ax = plt.subplots(figsize=(16,16))
plt.suptitle('Coherence Connectivity for Coupled (25 sec. epochs)',fontsize = 16)
#heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

im, cbar = heatmap(coupled, new_ch_names, new_ch_names, ax=ax,
                   cmap=plt.cm.Reds, cbarlabel="harvest [t/year]")

#%% contrast coupled - uncoupled
coupled = load_avg_matrix('coh','beta', 'Coupled', 'long', save = 1) 
uncoupled = load_avg_matrix('coh','beta', 'Uncoupled', 'long', save = 1) 
fig, ax = plt.subplots(figsize=(16,16))
plt.suptitle('Coherence Connectivity Coupled-Uncoupled (25 sec. epochs)',fontsize = 19,  y=0.95)
#heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

im, cbar = heatmap(coupled-uncoupled, new_ch_names, new_ch_names, ax=ax,
                   cmap=plt.cm.seismic, cbarlabel="Coherence Coefficient Difference")
plt.tight_layout(pad=0)

#%% contrast coupled - control
coupled = load_avg_matrix('coh','beta', 'Coupled', 'long', save = 1) 
control = load_avg_matrix('coh','beta', 'Control', 'long', save = 1) 
fig, ax = plt.subplots(figsize=(16,16))
plt.suptitle('Coherence Connectivity Coupled-Control (25 sec. epochs)',fontsize = 19,  y=0.95)
#heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

im, cbar = heatmap(coupled-control, new_ch_names, new_ch_names, ax=ax,
                   cmap=plt.cm.seismic, cbarlabel="Coherence Coefficient Difference")
plt.tight_layout(pad=0)

#%% contrast coupled - LF alpha
coupled_alpha_3s = load_avg_matrix('coh','alpha', 'Coupled', '3sec', save = 1) 
LF_alpha_3s = load_avg_matrix('coh','alpha', 'Leader-Follower', '3sec', save = 1) 
fig, ax = plt.subplots(figsize=(16,16))
plt.suptitle('Coherence for Alpha Coupled-LF (3 sec. epochs)',fontsize = 19,  y=0.95)
#heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

im, cbar = heatmap(coupled_alpha_3s-LF_alpha_3s, new_ch_names, new_ch_names, ax=ax,
                   cmap=plt.cm.seismic, cbarlabel="Coherence Coefficient Difference")
plt.tight_layout(pad=0)

#%% contrast coupled - LF beta
coupled_beta_3s = load_avg_matrix('coh','beta', 'Coupled', '3sec', save = 1) 
LF_beta_3s = load_avg_matrix('coh','beta', 'Leader-Follower', '3sec', save = 1) 
fig, ax = plt.subplots(figsize=(16,16))
plt.suptitle('Coherence for Beta Coupled-LF (3 sec. epochs)',fontsize = 19,  y=0.95)
#heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

im, cbar = heatmap(coupled_beta_3s-LF_beta_3s, new_ch_names, new_ch_names, ax=ax,
                   cmap=plt.cm.seismic, cbarlabel="Coherence Coefficient Difference")
plt.tight_layout(pad=0)

