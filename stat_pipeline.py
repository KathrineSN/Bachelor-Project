# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:54:56 2021

@author: kathr
"""

import os
import pickle
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
import hypyp

from permutation_test import *
#%%
epo1 = mne.read_epochs('epochs_a_short_10.fif')
epo2 = mne.read_epochs('epochs_b_short_10.fif')

# Permutation tests of ccorr values
#%% coupled vs. control
stat_ccorr_cou_con_alpha_s = permutation_test('ccorr', 'Coupled', 'Control', 'alpha', 'short')
stat_ccorr_cou_con_beta_s = permutation_test('ccorr', 'Coupled', 'Control', 'beta', 'short')
stat_ccorr_cou_con_theta_s = permutation_test('ccorr', 'Coupled', 'Control', 'theta', 'short')

stat_ccorr_cou_con_alpha = permutation_test('ccorr', 'Coupled', 'Control', 'alpha', 'long') # something significant...
stat_ccorr_cou_con_beta = permutation_test('ccorr', 'Coupled', 'Control', 'beta', 'long')
stat_ccorr_cou_con_theta = permutation_test('ccorr', 'Coupled', 'Control', 'theta', 'long')

#%% uncoupled vs. control 
stat_ccorr_unc_con_alpha_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'short')
stat_ccorr_unc_con_beta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'short')
stat_ccorr_unc_con_theta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'short')

stat_ccorr_unc_con_alpha = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'long') # something significant...
stat_ccorr_unc_con_beta = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'long')
stat_ccorr_unc_con_theta = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'long')

#%% coupled vs. LF
stat_ccorr_cou_LF_alpha_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'alpha', 'short')
stat_ccorr_cou_LF_beta_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'beta', 'short')
stat_ccorr_cou_LF_theta_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'theta', 'short')

stat_ccorr_cou_LF_alpha = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'alpha', 'long')
stat_ccorr_cou_LF_beta = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'beta', 'long')
stat_ccorr_cou_LF_theta = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'theta', 'long')

#%% coupled vs. uncoupled
stat_ccorr_cou_unc_alpha_s = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'alpha', 'short')
stat_ccorr_cou_unc_beta_s = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'beta', 'short')
stat_ccorr_cou_unc_theta_s = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'theta', 'short')

stat_ccorr_cou_unc_alpha = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'alpha', 'long')
stat_ccorr_cou_unc_beta = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'beta', 'long')
stat_ccorr_cou_unc_theta = permutation_test('ccorr', 'Uncoupled', 'Coupled', 'theta', 'long')

#%% Plot of significant clusters in uncoupled vs. control
#with open("con_names_order.txt", "rb") as fp:   # Unpickling
#    con_names = pickle.load(fp)
    
# Check if there are significant clusters
con = stat_ccorr_cou_con_theta
#con = stat_ccorr_unc_con_alpha
pv = con[2]
#pv = stat_ccorr_unc_con_alpha[2]

# Finding significant cluster
index = []
sig_pv = []

for i in range(len(pv)):
    if pv[i] < 0.05:
        sig_pv.append(pv[i])
        index.append(i)

t_values = con[0]
log = con[1]
for i in range(len(t_values)):
    if log[0][i] == False:
        t_values[i] = 0

# Number of connections in cluster
print(sum(log[0]))

# Matrix to plot significant clusters
m_init = np.zeros((64,64))
idx = 0
for i in range(64):
    for j in range(64):
        if i<=j:
            m_init[i,j] = t_values[idx]
            idx+=1

# Topomap
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
#plt.title('coupled-control alpha long')
plt.title('uncoupled-control alpha long')
viz.viz_2D_topomap_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=True)
viz.viz_3D_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=False)


#%% Potential plot for cluster permutation section

pv = stat_ccorr_unc_con_alpha[2]

# Finding significant cluster
index = []
sig_pv = []

for i in range(len(pv)):
    if pv[i] < 0.05:
        sig_pv.append(pv[i])
        index.append(i)
        
#%%
from mayavi.mlab import *

X = np.arange(0, 64, 1)
Y = np.arange(0, 64, 1)
Z = stat_ccorr_unc_con_theta[0]
Z2 = np.full(shape = (64,64), fill_value = 4.9646027437307145)

surf(X,Y,Z)
colorbar(orientation = 'vertical', nb_labels = 10)
surf(X,Y,Z2,color = (0.8,0.8,0.8))


#%%









