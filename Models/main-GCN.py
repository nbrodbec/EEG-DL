#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Hide the Configuration and Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import sys
sys.path.append('C:/Users/coffe/EEG-DL')

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse
from Models.Network.lib_for_GCN import GCN_Model, graph, coarsening
from Download_Raw_EEG_Data.physionet_dataset import extract_data_edf
from Models.Util.cm import cm_analysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from train_test_split import train_test_split

# Model Name
Model = 'Graph_Convolutional_Neural_Network'

# Clear all the stack and use GPU resources as much as possible
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Your Dataset Location, for example EEG-Motor-Movement-Imagery-Dataset
# The CSV file should be named as training_set.csv, training_label.csv, test_set.csv, and test_label.csv
DIR = 'DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/'
SAVE = 'Saved_Files/' + Model + '/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)

all = extract_data_edf('S001')
for i in range(2, 21):
	all = np.append(all, extract_data_edf('S{}'.format(str(i).zfill(3))), axis=0)

# Shuffled
# np.random.shuffle(all)

# Unshuffled

data = all[:, :64]
# data = data - data.mean(axis=1, keepdims=True)
labels = all[:, 64]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

# Read the Adjacency matrix
Adjacency_Matrix = pd.read_csv(DIR + 'Adjacency_Matrix.csv', header=None)
Adjacency_Matrix = np.array(Adjacency_Matrix).astype('float32')
Adjacency_Matrix = sparse.csr_matrix(Adjacency_Matrix)

# This is the coarsen levels, you can definitely change the level to observe the difference
graphs, perm = coarsening.coarsen(Adjacency_Matrix, levels=5, self_connections=False)
X_train = coarsening.perm_data(train_data, perm)
X_test  = coarsening.perm_data(test_data,  perm)

# Obtain the Graph Laplacian
L = [graph.laplacian(Adjacency_Matrix, normalized=True) for Adjacency_Matrix in graphs]

# Hyper-parameters
params = dict()
params['dir_name']       = Model
params['num_epochs']     = 10
params['batch_size']     = 1024
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu']  = 'b2relu'
params['pool']   = 'mpool1'

# Architecture.
params['F'] = [16, 32, 64, 128, 256, 512]  # Number of graph convolutional filters.
params['K'] = [2, 2, 2, 2, 2, 2]           # Polynomial orders.
params['p'] = [2, 2, 2, 2, 2, 2]           # Pooling sizes.
params['M'] = [5]                          # Output dimensionality of fully connected layers.

# Optimization.
params['regularization'] = 0.001     # L2 regularization
params['dropout']        = 0.50      # Dropout rate
params['learning_rate']  = 0.0001     # Learning rate; default = 0.000001
params['decay_rate']     = 1         # Learning rate Decay == 1 means no Decay
params['momentum']       = 0         # momentum == 0 means Use Adam Optimizer
params['decay_steps']    = np.shape(train_data)[0] / params['batch_size']


# Train model
model = GCN_Model.cgcnn(L, **params)
accuracy, loss, t_step = model.fit(X_train, train_labels, X_test, test_labels)

y_test = test_labels
y_pred = model.predict(X_test)

print(f'Test accuracy: {accuracy_score(y_test, y_pred)}')

classes_used = [0, 1, 2, 3, 4]
class_map = {0: 'Rest', 1: 'Left Fist', 2: 'Right Fist', 3: 'Both Fists', 4: 'Both Feet'}
cm_analysis(y_test, y_pred, labels=classes_used, ymap=class_map)