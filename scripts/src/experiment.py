#!/usr/bin/env python

import random

import click
import numpy as np
from sklearn import svm
from scipy.io import loadmat


class Experiment:
    def __init__(self):
        pass

    def svm_score(self, x_train, x_test, y_train, y_test, seed = 42):
        clf = svm.SVC(gamma = 'scale', class_weight="balanced", random_state = seed)
        clf.fit(x_train, y_train)

        score = clf.score(x_test, y_test)
        return score