# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:27:57 2021

@author: kathr
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pickle
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Bachelor-Project"
os.chdir(path)

# %% Cluster permutation test - define neighbors for pairs of electrodes
# To use the MNE function I need each group data in the format 3D array (n_obs, n_freq, n_connectivity_combination)
# And the 2 groups should be as elements in a list
# I should test each connectivity measurement separately

# First I have to define the "neighbors" for the connectivity combinations
# Definition: For the connection ch1 - ch2
# All spatial neighbors to ch1 and all spatial neighbors to ch2 and their connections are also considered neighbors!

epochs_a_s = mne.read_epochs('epochs_a_short_10.fif')
#epochs_b_s = mne.read_epochs('epochs_b_short_10.fif')

connectivity, ch_names = mne.channels.find_ch_adjacency(epochs_a_s.info, ch_type="eeg")

n_channels = len(ch_names)
ch_names = np.array(ch_names)

ch_names_new = []
connectivity_names_new = []
for i in range(n_channels):
    for j in range(n_channels):
        # Avoid connections between same electrode
        # Connectivity is symmetric, thus I will also avoid repeats, e.g. Fp1-Fp2, Fp2-Fp1
        
        if i > j:
            continue
        else:
        
            # Get the label for the connection
            ch_names_new.append(str(ch_names[i]+"-"+ch_names[j]))
            # Make temporary inner list for neighbors
            temp_con = []
            
            # Get number of neighbors (non-zeros) around j
            n_length = connectivity[j,:].getnnz()
            # Get index for the neighbors
            ch_idx = (connectivity[j,:].toarray() == 1)[0]
            # Get all the neighboring connections i to j neighbors
            for k in range(n_length):
                # The order ch-1-ch2 matter as I have i >= j and will need to use it for indexing as string
                ch_name1 = ch_names[i]
                ch_name2 = ch_names[ch_idx][k]
                # Remove connection to itself
                
                '''
                if ch_name1 == ch_name2:
                    continue
                else:
                    # If ch_name1 comes before ch_name2 then it should be written in front
                    if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                        temp_con.append(str(ch_name1+"-"+ch_name2))
                    else:
                        temp_con.append(str(ch_name2+"-"+ch_name1))
                
                '''
                 # If ch_name1 comes before ch_name2 then it should be written in front
                if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                    temp_con.append(str(ch_name1+"-"+ch_name2))
                else:
                    temp_con.append(str(ch_name2+"-"+ch_name1))
                
                        
            # Get number of neighbors (non-zeros) around i
            n_length2 = connectivity[i,:].getnnz()
            # Get index for the neighbors
            ch_idx2 = (connectivity[i,:].toarray() == 1)[0]
            # Get all the neighboring connections j to i neighbors
            for k2 in range(n_length2):
                # The order ch-1-ch2 matter as I have i >= j and will need to use it for indexing as string
                ch_name1 = ch_names[j]
                ch_name2 = ch_names[ch_idx2][k2]
                # Remove connection to itself
                
                '''
                if ch_name1 == ch_name2:
                    continue
                else:
                    # If ch_name1 comes before ch_name2 then it should be written in front
                    if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                        temp_con.append(str(ch_name1+"-"+ch_name2))
                    else:
                        temp_con.append(str(ch_name2+"-"+ch_name1))
                
                '''
                # If ch_name1 comes before ch_name2 then it should be written in front
                if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                    temp_con.append(str(ch_name1+"-"+ch_name2))
                else:
                    temp_con.append(str(ch_name2+"-"+ch_name1))
                
                
            # Get the "secondary" neighbors
            for k in range(n_length):
                for k2 in range(n_length2):
                    # The order matters as I have i >= j
                    ch_name1 = ch_names[ch_idx][k]
                    ch_name2 = ch_names[ch_idx2][k2]
                    #  Remove connection to itself
                    '''
                    if ch_name1 == ch_name2:
                        continue
                    else:
                        # If ch_name1 comes before ch_name2 then it should be written in front
                        if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                            temp_con.append(str(ch_name1+"-"+ch_name2))
                        else:
                            temp_con.append(str(ch_name2+"-"+ch_name1))
                    
                    '''
                    # If ch_name1 comes before ch_name2 then it should be written in front
                    if (np.where(ch_names == ch_name1)[0] < np.where(ch_names == ch_name2)[0])[0]:
                        temp_con.append(str(ch_name1+"-"+ch_name2))
                    else:
                        temp_con.append(str(ch_name2+"-"+ch_name1))
                    
                    
            # Remove redundant elements as I only need unique connections
            u_index = np.unique(temp_con, return_index=True)[1]
            # Get unique connections
            temp_unique = [temp_con[index] for index in sorted(u_index)]
            # Save the neighbor list
            connectivity_names_new.append(temp_unique)

# Check result shape
# How many times can I pick two out of the 64 = 2016
#n_ch_connections = scipy.special.comb(n_channels,2, exact=True, repetition=False)
n_ch_connections = 2080
assert len(ch_names_new) == n_ch_connections
# Convert connectivity names to sparse matrix
connectivity_new = np.zeros((n_ch_connections,n_ch_connections))

#connectivity_new = np.zeros((n_ch_connections,n_ch_connections))
for i in range(n_ch_connections):
    connectivity_new[i,:] = [ele in connectivity_names_new[i] for ele in ch_names_new]

# Convert to sparse matrix
connectivity_new = scipy.sparse.csr_matrix(connectivity_new)

# Visualize neighborhood connectivity matrix
plt.imshow(connectivity_new.toarray(), cmap="gray", origin="lower",
           interpolation="nearest")
plt.xlabel("Connectivity")
plt.ylabel("Connectivity")
plt.title("Between-connectivity adjacency")

# Test one neighbors for connection
chosen_connection = "Fp1-AF7"
connection_idx = ch_names_new.index(chosen_connection)
Neighbors_idx = connectivity_new.toarray()[connection_idx,:]
Neighbor_labels = np.array(ch_names_new)[np.where(Neighbors_idx==1)[0]]

#%% Saving the adjacency matrix

scipy.sparse.save_npz(path + '\\Adjacency\\adjacency.npz', connectivity_new)

#%% Save connection name order
with open("con_names_order.txt", "wb") as fp:   #Pickling
    pickle.dump(ch_names_new, fp)




