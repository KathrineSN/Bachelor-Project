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

# Plot it out
coupled = load_avg_matrix('coh','beta', 'Coupled', 'long', save = 1) 
fig, ax = plt.subplots()
plt.suptitle('coherence coupled beta long',fontsize = 16)
heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)

# Format
fig = plt.gcf()
fig.set_size_inches(10, 10)

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
