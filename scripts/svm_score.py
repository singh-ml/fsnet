#!/usr/bin/env python

from glob import glob 
from os.path import basename

from src.data_loader import Data
from src.experiment import Experiment

import numpy as np

tumors = ['breast', 'colorectal', 'gastric']

for tumor in tumors:
    d = Data(tumor + '/' + tumor + '.mat')
    for random_features in [False, True]:
        for feature_mat in glob(tumor + '/eta100/*.mat'): 
            d.select_features(feature_mat, random_features)

            for i in range(10):
                x_train, x_test, y_train, y_test = d.get_splits(0.5)
                score = Experiment().svm_score(x_train, x_test, y_train, y_test)
                print(tumor, basename(feature_mat), random_features, i, score)