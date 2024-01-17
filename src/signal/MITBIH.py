#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MITBIH.py

PyTorch dataloaders for MITHIB dataset

Author: Xiaomin Li, Texas State University
Date: 1/26/2023


TODOS:
* 
"""


#necessory import libraries

import os 
import sys 
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement
import random
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#class names and corresponding labels of the MITBIH dataset
cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
                                                'Unknown':3, 'Fusion Beats':4}
reverse_cls_dit = {0:'Non-Ectopic Beats', 1:'Superventrical Ectopic', 2:'Ventricular Beats', 3: 'Unknown', 4:'Fusion Beats'}



class mitbih_oneClass(Dataset):
    """
    A pytorch dataloader loads on class data from mithib_train dataset.
    Example Usage:
        class0 = mitbih_oneClass(class_id = 0)
        class1 = mitbih_oneClass(class_id = 1)
    """
    def __init__(self, filename='./mitbih_train.csv', reshape = True, class_id = 0):
        data_pd = pd.read_csv(filename, header=None)
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to only has 128 timesteps to match conv1d dim changes
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
        
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
class mitbih_twoClass(Dataset):
    """
    A pytorch dataloader loads two class data from mithib_train dataset.
    Example Usage:
        class0_1 = mitbih_twoClass(class_id1 = 0, class_id2 = 1)
        class1_2 = mitbih_twoClass(class_id1 = 1, class_id2 = 2)
    """
    def __init__(self, filename='./mitbih_train.csv', reshape = True, class_id1 = 0, class_id2 = 1):
        data_pd = pd.read_csv(filename, header=None)
        data_1 = data_pd[data_pd[187] == class_id1]
        data_2 = data_pd[data_pd[187] == class_id2]
    
        self.data_1 = data_1.iloc[:, :128].values
        self.labels_1 = data_1[187].values
        
        self.data_2 = data_2.iloc[:, :128].values
        self.labels_2 = data_2[187].values
        
        if reshape:
            self.data_1 = self.data_1.reshape(self.data_1.shape[0], 1, self.data_1.shape[1])
            self.data_2 = self.data_2.reshape(self.data_2.shape[0], 1, self.data_2.shape[1])
        
        print(f'Data shape of {reverse_cls_dit[class_id1]} instances = {self.data_1.shape}')
        print(f'Data shape of {reverse_cls_dit[class_id2]} instances = {self.data_2.shape}')
        
    def __len__(self):
        return min(len(self.labels_1), len(self.labels_2))
    
    def __getitem__(self, idx):
        return self.data_1[idx], self.labels_1[idx], self.data_2[idx], self.labels_2[idx]
    
    
class mitbih_masked(Dataset):
    def __init__(self, filename='./datasets/MITBIH/mitbih_train.csv', reshape = True, class_id = 0):
        data_pd = pd.read_csv(filename, header=None)
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to only has 128 timesteps to match conv1d dim changes
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
            
        n, ch, seq_len = self.data.shape[0], self.data.shape[1], self.data.shape[2]
        
        # random masked 20 timepoints in each data samples
        self.mask = np.full((n, ch, seq_len), True)
        for sample in range(len(self.mask)):
            for ch in range(len(self.mask[0])):
                self.mask[sample][ch][:20] = False
                np.random.shuffle(self.mask[sample][ch])
                
        self.cond_data = self.data * self.mask
        
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'org_data':self.data[idx], 'cond_data':self.cond_data[idx], 'label':self.labels[idx], 'idx':idx}
    
    

class mitbih_SR_HR(Dataset):
    def __init__(self, filename='./datasets/MITBIH/mitbih_train.csv', reshape = True, class_id = 0):
        data_pd = pd.read_csv(filename, header=None)
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to only has 128 timesteps to match conv1d dim changes
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
            
        n, ch, seq_len = self.data.shape[0], self.data.shape[1], self.data.shape[2]
        
        # random masked 30 timepoints in each data samples
        self.mask = np.full((n, ch, seq_len), True)
        for sample in range(len(self.mask)):
            for ch in range(len(self.mask[0])):
                self.mask[sample][ch][:30] = False
                np.random.shuffle(self.mask[sample][ch])
                
        self.cond_data = self.data * self.mask
        
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'HR':self.data[idx], 'SR':self.cond_data[idx], 'Labels':self.labels[idx], 'Index':idx}   
    
    
    
    
class mitbih_allClass(Dataset):
    def __init__(self, filename='./mitbih_train.csv', isBalanced = True, n_samples=20000, oneD=True):
        data_train = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        self.data_0 = data_train[data_train[187] == 0]
        self.data_1 = data_train[data_train[187] == 1]
        self.data_2 = data_train[data_train[187] == 2]
        self.data_3 = data_train[data_train[187] == 3]
        self.data_4 = data_train[data_train[187] == 4]
        
        if isBalanced:
            self.data_0_resample = resample(self.data_0, n_samples=n_samples, 
                               random_state=123, replace=True)
            self.data_1_resample = resample(self.data_1, n_samples=n_samples, 
                                       random_state=123, replace=True)
            self.data_2_resample = resample(self.data_2, n_samples=n_samples, 
                                       random_state=123, replace=True)
            self.data_3_resample = resample(self.data_3, n_samples=n_samples, 
                                       random_state=123, replace=True)
            self.data_4_resample = resample(self.data_4, n_samples=n_samples, 
                                       random_state=123, replace=True)

            train_dataset = pd.concat((self.data_0_resample, self.data_1_resample, 
                                      self.data_2_resample, self.data_3_resample, self.data_4_resample))
        else:
            train_dataset = pd.concat((self.data_0, self.data_1, 
                                      self.data_2, self.data_3, self.data_4))

        self.X_train = train_dataset.iloc[:, :128].values
        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])
        self.y_train = train_dataset[187].values
            
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        if isBalanced:
            print(f'The dataset including {len(self.data_0_resample)} class 0, {len(self.data_1_resample)} class 1, \
                  {len(self.data_2_resample)} class 2, {len(self.data_3_resample)} class 3, {len(self.data_4_resample)} class 4')
        else:
            print(f'The dataset including {len(self.data_0)} class 0, {len(self.data_1)} class 1, {len(self.data_2)} class 2, {len(self.data_3)} class 3, {len(self.data_4)} class 4')
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
class mitbih_denosing(Dataset):
    def __init__(self, dataroot='./mitbih_train.csv', train_mode = True, reshape = True, class_id = 0):
        if train_mode:
            data_pd = pd.read_csv(os.path.join(dataroot, 'mitbih_train.csv'), header=None)
        else:
            data_pd = pd.read_csv(os.path.join(dataroot, 'mitbih_test.csv'), header=None)
        
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to has 128 timesteps to match conv1d dim changes can evenely divided by 8
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
            
        n, ch, seq_len = self.data.shape[0], self.data.shape[1], self.data.shape[2]
    
        thermal_noise_level = 0.1
        thermal_noise = np.random.rand(n, ch, seq_len) * thermal_noise_level
        
        num_spikes_per_signal = np.random.randint(1, 4, n)  # Randomly select the number of spikes for each signal
        motion_artifacts = np.zeros_like(self.data)
        for i in range(n):
            spike_positions = np.random.randint(0, seq_len, num_spikes_per_signal[i])
            spike_amplitudes = np.random.uniform(0.3, 1, num_spikes_per_signal[i])
            for spike_pos, spike_amp in zip(spike_positions, spike_amplitudes):
                motion_artifacts[i, 0, spike_pos:spike_pos + 5] = spike_amp

        self.motion_artifacts = motion_artifacts
        self.cond_data = self.data + thermal_noise + motion_artifacts
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'ORG':self.data[idx], 'COND':self.cond_data[idx], 'Labels':self.labels[idx], 'Index':idx}  
    
    
class mitbih_imputation(Dataset):
    def __init__(self, dataroot='./mitbih_train.csv', train_mode = True, reshape = True, class_id = 0):
        if train_mode:
            data_pd = pd.read_csv(os.path.join(dataroot, 'mitbih_train.csv'), header=None)
        else:
            data_pd = pd.read_csv(os.path.join(dataroot, 'mitbih_test.csv'), header=None)
        
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to has 128 timesteps to match conv1d dim changes can evenely divided by 8
        self.labels = data[187].values
        self.cond_data = self.data.copy()
            
        num_signals, signal_length = self.data.shape
        
        # Calculate the number of groups to mask (approximately 10% of the data points)
        group_size_mean = (5 + 10) / 2
        num_masked_groups = int(num_signals * signal_length * 0.1 / group_size_mean)

        # Generate random start indices for each group
        start_indices = np.random.randint(0, signal_length - 10, num_masked_groups)

        # Calculate the size of each group (5-10 timepoints)
        group_sizes = np.random.randint(5, 11, num_masked_groups)

        # Create a mask numpy array with the same shape as the clean signals
        mask_array = np.zeros((num_signals, signal_length), dtype=bool)

        # Set the specified groups of indices to 1 (True)
        for i in range(num_signals):
            for start_idx, size in zip(start_indices[i::num_signals], group_sizes[i::num_signals]):
                mask_array[i, start_idx:start_idx + size] = True

                
        self.mask = mask_array.copy()
        self.cond_data[mask_array] = 0  # Or any other value you want to replace the masked data points with
        
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
            self.cond_data = self.cond_data.reshape(self.cond_data.shape[0], 1, self.cond_data.shape[1])
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'ORG':self.data[idx], 'COND':self.cond_data[idx], 'Labels':self.labels[idx], 'Index':idx}  
    
    
class mitbih_augData(Dataset):
    def __init__(self, filename='./mitbih_train.csv', n_samples=20000, target_class = 0):
        data_train = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0].iloc[:, :128].values
        data_0 = np.expand_dims(data_0, axis=1)
        data_1 = data_train[data_train[187] == 1].iloc[:, :128].values
        data_1 = np.expand_dims(data_1, axis=1)
        data_2 = data_train[data_train[187] == 2].iloc[:, :128].values
        data_2 = np.expand_dims(data_2, axis=1)
        data_3 = data_train[data_train[187] == 3].iloc[:, :128].values
        data_3 = np.expand_dims(data_3, axis=1)
        data_4 = data_train[data_train[187] == 4].iloc[:, :128].values
        data_4 = np.expand_dims(data_4, axis=1)
        
        data_0_resample = resample(data_0, n_samples=n_samples, 
                           random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples, 
                                   random_state=123, replace=True)

        
        org_dataset = np.concatenate((data_0_resample, data_1_resample, 
                                      data_2_resample, data_3_resample, data_4_resample), axis=1)
        print(org_dataset.shape) #[n_samples, 5, 187]
        

        self.org_train = np.expand_dims(org_dataset, axis=1)
        print(f'org_train shape is {self.org_train.shape}') #[n_samples, 1, 5, 187]
        
        tag_data = data_train[data_train[187] == target_class].iloc[:, :-1].values
        tag_data = np.expand_dims(tag_data, axis=1)
        tag_data_resample = resample(tag_data, n_samples=n_samples, 
                           random_state=123, replace=True)
        self.tag_train = np.expand_dims(tag_data_resample, axis=1)
        print(f'tag_train shape is {self.tag_train.shape}') #[n_samples, 1, 1, 187]
        
    def __len__(self):
        return len(self.tag_train)
    def __getitem__(self, idx):
        return self.org_train[idx], self.tag_train[idx]

    
    
    
#synthetic heartbeat signal dataloader
class syn_mitbih(Dataset):
    def __init__(self, data_folder = './synthetic/synthetic_data/', n_samples = 1000):
        
        # shape = n_samples, 1, seq_len
        self.syn_0 = np.load(os.path.join(data_folder, 'syn_MITBIH_class0.npy'))
        self.syn_1 = np.load(os.path.join(data_folder, 'syn_MITBIH_class1.npy'))
        self.syn_2 = np.load(os.path.join(data_folder, 'syn_MITBIH_class2.npy'))
        self.syn_3 = np.load(os.path.join(data_folder, 'syn_MITBIH_class3.npy'))
        self.syn_4 = np.load(os.path.join(data_folder, 'syn_MITBIH_class4.npy'))
        
        self.syn_0 = resample(self.syn_0, n_samples=n_samples, random_state=123, replace=True)
        self.syn_1 = resample(self.syn_1, n_samples=n_samples, random_state=123, replace=True)
        self.syn_2 = resample(self.syn_2, n_samples=n_samples, random_state=123, replace=True)
        self.syn_3 = resample(self.syn_3, n_samples=n_samples, random_state=123, replace=True)
        self.syn_4 = resample(self.syn_4, n_samples=n_samples, random_state=123, replace=True)
        
        self.data = np.concatenate((self.syn_0, self.syn_1, self.syn_2, self.syn_3, self.syn_4), axis = 0)
        self.labels = np.concatenate((np.array([0]*len(self.syn_0)), np.array([1]*len(self.syn_0)), np.array([2]*len(self.syn_0)), np.array([3]*len(self.syn_0)), np.array([4]*len(self.syn_0))))
            
        print(f'data shape is {self.data.shape}')
        print(f'labels shape is {self.labels.shape}')
        print(f'The dataset including {len(self.syn_0)} class 0, {len(self.syn_0)} class 1, {len(self.syn_0)} class 2, {len(self.syn_0)} class 3, {len(self.syn_0)} class 4')
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class mixed_mitbih(Dataset):
    def __init__(self, real_samples = 200, syn_samples = 800):
        syn_ecg = syn_mitbih(n_samples = syn_samples)
        real_ecg = mitbih_allClass(filename='./datasets/MITBIH/mitbih_train.csv', isBalanced=False)
        
        real_0 = resample(real_ecg.data_0, n_samples=real_samples, random_state=123, replace=True)
        real_0 = real_0.iloc[:, :128].values
        real_0 = real_0.reshape(real_0.shape[0], 1, real_0.shape[1])

        real_1 = resample(real_ecg.data_1, n_samples=real_samples, random_state=123, replace=True)
        real_1 = real_1.iloc[:, :128].values
        real_1 = real_1.reshape(real_1.shape[0], 1, real_1.shape[1])

        real_2 = resample(real_ecg.data_2, n_samples=real_samples, random_state=123, replace=True)
        real_2 = real_2.iloc[:, :128].values
        real_2 = real_2.reshape(real_2.shape[0], 1, real_2.shape[1])

        real_3 = resample(real_ecg.data_3, n_samples=real_samples, random_state=123, replace=True)
        real_3 = real_3.iloc[:, :128].values
        real_3 = real_3.reshape(real_3.shape[0], 1, real_3.shape[1])

        real_4 = resample(real_ecg.data_4, n_samples=real_samples, random_state=123, replace=True)
        real_4 = real_4.iloc[:, :128].values
        real_4 = real_4.reshape(real_4.shape[0], 1, real_4.shape[1])

        
        self.data = np.concatenate((real_0, real_1, real_2, real_3, real_4, syn_ecg.syn_0, syn_ecg.syn_1, syn_ecg.syn_2, syn_ecg.syn_3, syn_ecg.syn_4), axis = 0)
        self.labels = np.concatenate(([0]*len(real_0), [1]*len(real_1), [2]*len(real_2), [3]*len(real_3), [4]*len(real_4), \
                                     [0]*len(syn_ecg.syn_0), [1]*len(syn_ecg.syn_1), [2]*len(syn_ecg.syn_2), [3]*len(syn_ecg.syn_3), [4]*len(syn_ecg.syn_4)), axis = 0)
        
        print(f'data shape is {self.data.shape}')
        print(f'labels shape is {self.labels.shape}')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
    
# #################################
# # mixed_mitbih with unbalanced real training data
# class mixed_mitbih_imbalanced(Dataset):
#     def __init__(self, sample_rate=0.1, random_seed = 123, syn_samples=1000):
#         syn_ecg = syn_mitbih(n_samples=syn_samples, reshape=True)
# #         real_ecg = mitbih_train_imbalanced(sample_rate=0.1, random_seed = 123, oneD=True)
#         real_ecg = mitbih_train_balanced(sample_rate=0.1, n_samples=1000, random_seed = 123, oneD=True)
        
#         self.data = np.concatenate((syn_ecg.data, real_ecg.X_train), axis = 0)
#         self.labels = np.concatenate((syn_ecg.labels, real_ecg.y_train), axis = 0)
        
#         print(f'data shape is {self.data.shape}')
#         print(f'labels shape is {self.labels.shape}')
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]


def main():
#     class0 = mitbih_oneClass(class_id = 0)
#     class_0_1 = mitbih_twoClass(class_id1 = 0, class_id2 = 1)
#     data = mitbih_allClass(isBalanced = True, n_samples=2000)
#     data = mitbih_allClass(isBalanced = False)
    
#     mixData = mitbih_augData(n_samples=2000)
#     for i, (org_data, tag_data) in enumerate(mixData):
#         print(org_data.shape)
#         print(tag_data.shape)
#         if i > 3:
#             break
    mixed_mitbih()
        
if __name__ == "__main__":
    main()