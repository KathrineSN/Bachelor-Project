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
    cbar.ax.tick_params(labelsize=10)

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



#%%

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(coupled.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(coupled.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(new_ch_names, minor=False)
ax.set_yticklabels(new_ch_names, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

#%% Coherence Connectivity for Uncoupled (25 sec.)

uncoupled = load_avg_matrix('coh','beta', 'Uncoupled', 'long', save = 1) 
fig, ax = plt.subplots()
plt.suptitle('Coherence Connectivity for Uncoupled (25 sec. epochs)',fontsize = 16)
heatmap = ax.pcolor(uncoupled, cmap=plt.cm.Reds, alpha=0.8)

# Format
fig = plt.gcf()
fig.set_size_inches(10, 10)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(uncoupled.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(uncoupled.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(new_ch_names, minor=False)
ax.set_yticklabels(new_ch_names, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

#%% Coherence contrast between coupled and uncoupled
contrast = coupled-uncoupled
fig, ax = plt.subplots()
plt.suptitle('Coherence Connectivity Coupled-Uncoupled (25 sec. epochs)',fontsize = 16)
heatmap = ax.pcolor(contrast, cmap=plt.cm.seismic, alpha=0.8)

# Format
fig = plt.gcf()
fig.set_size_inches(10, 10)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(contrast.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(contrast.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

montage = mne.channels.make_standard_montage("biosemi64")
new_ch_names = montage.ch_names

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(new_ch_names, minor=False)
ax.set_yticklabels(new_ch_names, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

#%%

