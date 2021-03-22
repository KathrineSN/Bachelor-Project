# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:54:56 2021

@author: kathr
"""

import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

from permutation_test import *

# Permutation tests of ccorr values
#%% coupled vs. control
stat_ccorr_coupled_alpha_s = permutation_test('ccorr', 'Coupled', 'Control', 'alpha', 'short')
stat_ccorr_coupled_beta_s = permutation_test('ccorr', 'Coupled', 'Control', 'beta', 'short')
stat_ccorr_coupled_theta_s = permutation_test('ccorr', 'Coupled', 'Control', 'theta', 'short')

stat_ccorr_coupled_alpha = permutation_test('ccorr', 'Coupled', 'Control', 'alpha', 'long')
stat_ccorr_coupled_beta = permutation_test('ccorr', 'Coupled', 'Control', 'beta', 'long')
stat_ccorr_coupled_theta = permutation_test('ccorr', 'Coupled', 'Control', 'theta', 'long')

#%% uncoupled vs. control 
stat_ccorr_uncoupled_alpha_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'short')
stat_ccorr_uncoupled_beta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'short')
stat_ccorr_uncoupled_theta_s = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'short')

stat_ccorr_uncoupled_alpha = permutation_test('ccorr', 'Uncoupled', 'Control', 'alpha', 'long') # something significant...
stat_ccorr_uncoupled_beta = permutation_test('ccorr', 'Uncoupled', 'Control', 'beta', 'long')
stat_ccorr_uncoupled_theta = permutation_test('ccorr', 'Uncoupled', 'Control', 'theta', 'long')

#%% LF vs control
stat_ccorr_LF_alpha_s = permutation_test('ccorr', 'Leader-Follower', 'Control', 'alpha', 'short')
stat_ccorr_LF_beta_s = permutation_test('ccorr', 'Leader-Follower', 'Control', 'beta', 'short')
stat_ccorr_LF_theta_s = permutation_test('ccorr', 'Leader-Follower', 'Control', 'theta', 'short')

stat_ccorr_LF_alpha = permutation_test('ccorr', 'Leader-Follower', 'Control', 'alpha', 'long')
stat_ccorr_LF_beta = permutation_test('ccorr', 'Leader-Follower', 'Control', 'beta', 'long')
stat_ccorr_LF_theta = permutation_test('ccorr', 'Leader-Follower', 'Control', 'theta', 'long')














