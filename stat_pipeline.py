# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:54:56 2021

@author: kathr
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
import hypyp
import mne

from permutation_test import permutation_test
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

stat_ccorr_cou_con_alpha_3s = permutation_test('ccorr', 'Coupled', 'Control', 'alpha', '3sec') 
stat_ccorr_cou_con_beta_3s = permutation_test('ccorr', 'Coupled', 'Control', 'beta', '3sec')
stat_ccorr_cou_con_theta_3s = permutation_test('ccorr', 'Coupled', 'Control', 'theta', '3sec')

#%% uncoupled vs. control 
stat_ccorr_unc_con_alpha_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'short')
stat_ccorr_unc_con_beta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'short')
stat_ccorr_unc_con_theta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'short')

stat_ccorr_unc_con_alpha = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'long') # something significant...
stat_ccorr_unc_con_beta = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'long')
stat_ccorr_unc_con_theta = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'long')

stat_ccorr_unc_con_alpha_3s = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', '3sec')
stat_ccorr_unc_con_beta_3s = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', '3sec')
stat_ccorr_unc_con_theta_3s = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', '3sec')

#%% coupled vs. LF
stat_ccorr_cou_LF_alpha_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'alpha', 'short')
stat_ccorr_cou_LF_beta_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'beta', 'short')
stat_ccorr_cou_LF_theta_s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'theta', 'short')

stat_ccorr_cou_LF_alpha = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'alpha', 'long')
stat_ccorr_cou_LF_beta = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'beta', 'long')
stat_ccorr_cou_LF_theta = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'theta', 'long')

stat_ccorr_cou_LF_alpha_3s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'alpha', '3sec')
stat_ccorr_cou_LF_beta_3s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'beta', '3sec')
stat_ccorr_cou_LF_theta_3s = permutation_test('ccorr', 'Leader-Follower', 'Coupled', 'theta', '3sec')

#%% coupled vs. uncoupled
stat_ccorr_cou_unc_alpha_s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'alpha', 'short')
stat_ccorr_cou_unc_beta_s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'beta', 'short')
stat_ccorr_cou_unc_theta_s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'theta', 'short')

stat_ccorr_cou_unc_alpha = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'alpha', 'long')
stat_ccorr_cou_unc_beta = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'beta', 'long')
stat_ccorr_cou_unc_theta = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'theta', 'long')

stat_ccorr_cou_unc_alpha_3s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'alpha', '3sec')
stat_ccorr_cou_unc_beta_3s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'beta', '3sec')
stat_ccorr_cou_unc_theta_3s = permutation_test('ccorr', 'Coupled', 'Uncoupled', 'theta', '3sec')

#%% control vs. resting
'''
stat_ccorr_con_res_alpha_s = permutation_test('ccorr', 'Resting', 'Control', 'alpha', 'short')
stat_ccorr_con_res_beta_s = permutation_test('ccorr', 'Resting', 'Control', 'beta', 'short')
stat_ccorr_con_res_theta_s = permutation_test('ccorr', 'Resting', 'Control', 'theta', 'short')
#%%
stat_ccorr_con_res_alpha = permutation_test('ccorr', 'Resting', 'Control', 'alpha', 'long')
stat_ccorr_con_res_beta = permutation_test('ccorr', 'Resting', 'Control', 'beta', 'long')
stat_ccorr_con_res_theta = permutation_test('ccorr', 'Resting', 'Control', 'theta', 'long')
'''
#%% Plot of significant clusters in uncoupled vs. control
'''
#with open("con_names_order.txt", "rb") as fp:   # Unpickling
#    con_names = pickle.load(fp)
    
# Check if there are significant clusters
con = stat_ccorr_cou_unc_beta
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
#plt.title('uncoupled-control alpha long')
hypyp.viz.viz_2D_topomap_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=True)
#hypyp.viz.viz_3D_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=False)
plt.title('coupled-uncoupled beta long')


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
'''

# Permutation tests of coh values
#%% coupled vs. control
stat_coh_cou_con_alpha_s = permutation_test('coh', 'Coupled', 'Control', 'alpha', 'short')
stat_coh_cou_con_beta_s = permutation_test('coh', 'Coupled', 'Control', 'beta', 'short')
stat_coh_cou_con_theta_s = permutation_test('coh', 'Coupled', 'Control', 'theta', 'short')

stat_coh_cou_con_alpha = permutation_test('coh', 'Coupled', 'Control', 'alpha', 'long') 
stat_coh_cou_con_beta = permutation_test('coh', 'Coupled', 'Control', 'beta', 'long') #sig
stat_coh_cou_con_theta = permutation_test('coh', 'Coupled', 'Control', 'theta', 'long')

stat_coh_cou_con_alpha_3s = permutation_test('coh', 'Coupled', 'Control', 'alpha', '3sec')
stat_coh_cou_con_beta_3s = permutation_test('coh', 'Coupled', 'Control', 'beta', '3sec')
stat_coh_cou_con_theta_3s = permutation_test('coh', 'Coupled', 'Control', 'theta', '3sec')

#%% uncoupled vs. control 
stat_coh_unc_con_alpha_s = permutation_test('coh', 'Uncoupled', 'Control', 'alpha', 'short')
stat_coh_unc_con_beta_s = permutation_test('coh', 'Uncoupled', 'Control', 'beta', 'short')
stat_coh_unc_con_theta_s = permutation_test('coh', 'Uncoupled', 'Control', 'theta', 'short')

stat_coh_unc_con_alpha = permutation_test('coh', 'Uncoupled', 'Control', 'alpha', 'long') 
stat_coh_unc_con_beta = permutation_test('coh', 'Uncoupled', 'Control', 'beta', 'long')
stat_coh_unc_con_theta = permutation_test('coh', 'Uncoupled', 'Control', 'theta', 'long')

stat_coh_unc_con_alpha_3s = permutation_test('coh', 'Uncoupled', 'Control', 'alpha', '3sec') 
stat_coh_unc_con_beta_3s = permutation_test('coh', 'Uncoupled', 'Control', 'beta', '3sec')
stat_coh_unc_con_theta_3s = permutation_test('coh', 'Uncoupled', 'Control', 'theta', '3sec')

#%% coupled vs. LF
stat_coh_cou_LF_alpha_s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'alpha', 'short')
stat_coh_cou_LF_beta_s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'beta', 'short')
stat_coh_cou_LF_theta_s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'theta', 'short')

stat_coh_cou_LF_alpha = permutation_test('coh', 'Coupled', 'Leader-Follower', 'alpha', 'long')
stat_coh_cou_LF_beta = permutation_test('coh', 'Coupled', 'Leader-Follower', 'beta', 'long')
stat_coh_cou_LF_theta = permutation_test('coh', 'Coupled', 'Leader-Follower', 'theta', 'long')

stat_coh_cou_LF_alpha_3s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'alpha', '3sec') #sig
stat_coh_cou_LF_beta_3s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'beta', '3sec') #sig
stat_coh_cou_LF_theta_3s = permutation_test('coh', 'Coupled', 'Leader-Follower', 'theta', '3sec')

#%% coupled vs. uncoupled
stat_coh_cou_unc_alpha_s = permutation_test('coh', 'Uncoupled', 'Coupled', 'alpha', 'short')
stat_coh_cou_unc_beta_s = permutation_test('coh', 'Uncoupled', 'Coupled', 'beta', 'short')
stat_coh_cou_unc_theta_s = permutation_test('coh', 'Uncoupled', 'Coupled', 'theta', 'short')

stat_coh_cou_unc_alpha = permutation_test('coh', 'Coupled', 'Uncoupled', 'alpha', 'long')
#%%
import random
random.seed(222)
stat_coh_cou_unc_beta_reversed = permutation_test('coh', 'Uncoupled', 'Coupled', 'beta', 'long') #sig
stat_coh_cou_unc_beta = permutation_test('coh', 'Coupled', 'Uncoupled', 'beta', 'long') #sig
#%%
stat_coh_cou_unc_theta = permutation_test('coh', 'Coupled', 'Uncoupled', 'theta', 'long')

stat_coh_cou_unc_alpha_3s = permutation_test('coh', 'Uncoupled', 'Coupled', 'alpha', '3sec')
stat_coh_cou_unc_beta_3s = permutation_test('coh', 'Uncoupled', 'Coupled', 'beta', '3sec')
stat_coh_cou_unc_theta_3s = permutation_test('coh', 'Uncoupled', 'Coupled', 'theta', '3sec')

#%% control vs. resting
'''
stat_coh_con_res_alpha_s = permutation_test('coh', 'Resting', 'Control', 'alpha', 'short')
stat_coh_con_res_beta_s = permutation_test('coh',  'Resting', 'Control', 'beta', 'short')
stat_coh_con_res_theta_s = permutation_test('coh', 'Resting', 'Control', 'theta', 'short')
#%%
stat_coh_con_res_alpha = permutation_test('coh', 'Resting', 'Control', 'alpha', 'long')
stat_coh_con_res_beta = permutation_test('coh', 'Resting', 'Control', 'beta', 'long')
stat_coh_con_res_theta = permutation_test('coh', 'Resting', 'Control', 'theta', 'long')
'''

#%% Plot of significant clusters with coh
#with open("con_names_order.txt", "rb") as fp:   # Unpickling
#    con_names = pickle.load(fp)
    
# Check if there are significant clusters
#con = stat_coh_cou_LF_beta_3s[0]
con = stat_coh_cou_LF_alpha_3s[0]
#con = stat_coh_cou_con_beta[0]
#con = stat_coh_cou_unc_beta[0]
#con = stat_coh_cou_unc_beta_reversed[0]
pv = con[2].copy()
#pv = stat_ccorr_unc_con_alpha[2]

# Finding significant cluster
index = []
sig_pv = []

for i in range(len(pv)):
    if pv[i] < 0.05:
        sig_pv.append(pv[i])
        index.append(i)

t_values = con[0].copy()
log = con[1].copy()
for i in range(len(t_values)):
    if log[index[0]][i] == False:
        t_values[i] = 0

# Number of connections in cluster
print(sum(log[index[0]]))

# Matrix to plot significant clusters
m_init = np.zeros((64,64))
idx = 0
for i in range(64):
    for j in range(64):
        if i<=j:
            m_init[i,j] = t_values[idx]
            idx+=1

#%%
# Topomap
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
#plt.title('uncoupled-control alpha long')
#fig = plt.Figure(figsize=(10,10))
hypyp.viz.viz_2D_topomap_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=True)
#plt.title('Coupled - Uncoupled (beta, 25 sec. epochs)')
plt.title('Coupled - Control (beta, 25 sec. epochs)')
#plt.title('Coupled - Leader-Follower (alpha, 3 sec. epochs)')
#plt.title('Coupled - Leader-Follower (beta, 3 sec. epochs)')

#NB! click tight layout before saving

#%%
hypyp.viz.viz_3D_inter(epo1, epo2, m_init, threshold='auto', steps=10, lab=False)
#plt.title('Coupled - Uncoupled (beta, 25 sec. epochs)')
#plt.title('Coupled - Control (beta, 25 sec. epochs)')
#plt.title('Coupled - Leader-Follower (alpha, 3 sec. epochs)')
#plt.title('Coupled - Leader-Follower (beta, 3 sec. epochs)')



#%% Prep for barplot
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)
from con_functions import load_avg_matrix
m_logic = m_init != 0
uncoupled = load_avg_matrix('coh','beta', 'Uncoupled', 'long', save = 0)
coupled = load_avg_matrix('coh','beta', 'Coupled', 'long', save = 0) 
control = load_avg_matrix('coh','beta', 'Control', 'long', save = 0) 
coupled_3s_alpha = load_avg_matrix('coh','alpha', 'Coupled', '3sec', save = 0) 
LF_3s_alpha = load_avg_matrix('coh','alpha', 'Leader-Follower', '3sec', save = 0) 
coupled_3s_beta = load_avg_matrix('coh','beta', 'Coupled', '3sec', save = 0) 
LF_3s_beta = load_avg_matrix('coh','beta', 'Leader-Follower', '3sec', save = 0) 

#%% Setting font sizes
import matplotlib.pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#%% Barplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

uncoupled_select = uncoupled[m_logic]
coupled_select = coupled[m_logic]
con_values = np.concatenate((coupled_select,uncoupled_select), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 55),np.repeat('Uncoupled', 55)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
plt.subplots_adjust(left = 0.18)
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Uncoupled')

plt.show()

#%% Plot with control and coupled
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

coupled_select = coupled[m_logic]
control_select = control[m_logic]
con_values = np.concatenate((coupled_select,control_select), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 62),np.repeat('Control', 62)), axis = 0)
d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Control')

plt.show()


#%% Plot with alpha Leader-Follower
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

coupled_3s_alpha_select = coupled_3s_alpha[m_logic]
LF_3s_alpha_select = LF_3s_alpha[m_logic]
con_values = np.concatenate((coupled_3s_alpha_select, LF_3s_alpha_select), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 64),np.repeat('Leader-Follower', 64)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Leader-Follower')

plt.show()

#%% Plot with beta Leader-Follower
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

coupled_3s_beta_select = coupled_3s_beta[m_logic]
LF_3s_beta_select = LF_3s_beta[m_logic]
con_values = np.concatenate((coupled_3s_beta_select, LF_3s_beta_select), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 89),np.repeat('Leader-Follower', 89)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Leader-Follower')

plt.show()

#%% Barplot with participants for Coupled vs. Uncoupled
import pandas as pd
import seaborn as sns
pairs = ['pair003','pair004','pair005','pair007','pair009','pair0010','pair0011','pair0012','pair0014','pair0016','pair0017','pair0018']
cou_means = []
unc_means = []
for i in pairs:
    # Loading coupled
    mat_cou = np.load('Connectivity matrices/coh/coh_'+i+'_beta_Coupled_long.npy')
    chan_picks_cou = mat_cou[m_logic]
    mean_val_cou = np.mean(chan_picks_cou)
    cou_means.append(mean_val_cou)
    
    # Loading uncoupled
    mat_unc = np.load('Connectivity matrices/coh/coh_'+i+'_beta_Uncoupled_long.npy')
    chan_picks_unc = mat_unc[m_logic]
    mean_val_unc = np.mean(chan_picks_unc)
    unc_means.append(mean_val_unc)
    
con_values = np.concatenate((np.array(cou_means), np.array(unc_means)), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 12),np.repeat('Uncoupled', 12)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Uncoupled per. participant')

plt.show()

#%% Barplot with participants for Coupled vs. Control
import pandas as pd
import seaborn as sns    
pairs = ['pair003','pair004','pair005','pair007','pair009','pair0010','pair0011','pair0012','pair0014','pair0016','pair0017','pair0018']
cou_means = []
con_means = []
for i in pairs:
    # Loading coupled
    mat_cou = np.load('Connectivity matrices/coh/coh_'+i+'_beta_Coupled_long.npy')
    chan_picks_cou = mat_cou[m_logic]
    mean_val_cou = np.mean(chan_picks_cou)
    cou_means.append(mean_val_cou)
    
    # Loading uncoupled
    mat_con = np.load('Connectivity matrices/coh/coh_'+i+'_beta_Control_long.npy')
    chan_picks_con = mat_con[m_logic]
    mean_val_con = np.mean(chan_picks_con)
    con_means.append(mean_val_con)
    
con_values = np.concatenate((np.array(cou_means), np.array(con_means)), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 12),np.repeat('Control', 12)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Control pr. participant')

plt.show()

#%% Barplot with participants for Coupled vs. LF (beta)
import pandas as pd
import seaborn as sns    
pairs = ['pair003','pair004','pair005','pair007','pair009','pair0010','pair0011','pair0012','pair0014','pair0016','pair0017','pair0018']
cou_means = []
LF_means = []
for i in pairs:
    # Loading coupled
    mat_cou = np.load('Connectivity matrices/coh/coh_'+i+'__beta_Coupled_3sec.npy')
    chan_picks_cou = mat_cou[m_logic]
    mean_val_cou = np.mean(chan_picks_cou)
    cou_means.append(mean_val_cou)
    
    # Loading uncoupled
    mat_LF = np.load('Connectivity matrices/coh/coh_'+i+'__beta_Leader-Follower_3sec.npy')
    chan_picks_LF = mat_LF[m_logic]
    mean_val_LF = np.mean(chan_picks_LF)
    LF_means.append(mean_val_LF)
    
con_values = np.concatenate((np.array(cou_means), np.array(LF_means)), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 12),np.repeat('Leader-Follower', 12)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Leader-Follower pr. participant (beta)')

plt.show()

#%% Barplot with participants for Coupled vs. LF (alpha)
import pandas as pd
import seaborn as sns    
pairs = ['pair003','pair004','pair005','pair007','pair009','pair0010','pair0011','pair0012','pair0014','pair0016','pair0017','pair0018']
cou_means = []
LF_means = []
for i in pairs:
    # Loading coupled
    mat_cou = np.load('Connectivity matrices/coh/coh_'+i+'__alpha_Coupled_3sec.npy')
    chan_picks_cou = mat_cou[m_logic]
    mean_val_cou = np.mean(chan_picks_cou)
    cou_means.append(mean_val_cou)
    
    # Loading uncoupled
    mat_LF = np.load('Connectivity matrices/coh/coh_'+i+'__alpha_Leader-Follower_3sec.npy')
    chan_picks_LF = mat_LF[m_logic]
    mean_val_LF = np.mean(chan_picks_LF)
    LF_means.append(mean_val_LF)
    
con_values = np.concatenate((np.array(cou_means), np.array(LF_means)), axis = 0)
con_names = np.concatenate((np.repeat('Coupled', 12),np.repeat('Leader-Follower', 12)), axis = 0)

d = {'Coherence Coefficient': con_values, 'Condition': con_names}
df = pd.DataFrame(data = d)

#tips = sns.load_dataset("tips")
plt.close('all')
fig, ax = plt.subplots(1,1,figsize = (5,7))
sns.barplot(x='Condition', y='Coherence Coefficient', data=df, capsize=.1, ci="sd", ax = ax)
sns.swarmplot(x='Condition', y='Coherence Coefficient', data=df, color="0", alpha=.35, ax = ax)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
plt.title('Barplot of Coupled and Leader-Follower pr. participant (alpha)')

plt.show()


