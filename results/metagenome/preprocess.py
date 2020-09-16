#!/usr/bin/env python

import numpy as np
import scipy.io as scio
from sklearn.utils import resample

y = np.load('HMP2_Metagenome_keggko_summary_Y_filter.npy')
y = np.transpose(y)

# make feature matrix
features = np.load('HMP2_Metagenome_keggko_summary_X_filter.npy')
cov = np.load('HMP2_Metagenome_keggko_summary_cov_filter.npy')
x = np.concatenate((np.transpose(features), cov), axis = 1)

# upsample minority class
unique_elements, counts_elements = np.unique(y, return_counts=True)
idx_minority = (y == unique_elements[np.argmin(counts_elements)])[:,0]

x_minority, y_minority = resample(x[idx_minority,:], y[idx_minority,:],
                                  replace = True,
                                  n_samples = np.max(counts_elements),
                                  random_state = 42)
x_majority = x[np.logical_not(idx_minority),:]
y_majority = y[np.logical_not(idx_minority),:]

ds = {'X' : x, 'Y' : y }

scio.savemat('metagenome.mat', ds)
