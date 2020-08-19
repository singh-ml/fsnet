#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


class Data:
    def __init__(self, mat_data):
        
        data = loadmat(mat_data)
        x = data['X']
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)

        self.x_raw = self.x = x[idx, :]

        y = data['Y']
        y = y[idx]
        self.y = y

    def select_features(self, mat_feats, random = False):
        features = loadmat(mat_feats)
        features = features['indices'][0]
        if random:
            D = self.x.shape[1]
            features = np.random.choice(D, len(features), replace = False)

        self.x = self.x_raw[:, features]

    def get_splits(self, ratio = 0.9):
        N = self.x.shape[0]
        samples = np.random.choice(N, round(ratio * N), replace = False)
        
        x_train = self.x[samples,:]
        y_train = self.y[samples]

        x_test = np.delete(self.x, samples, 0)
        y_test = np.delete(self.y, samples, 0)

        return x_train, x_test, y_train, y_test