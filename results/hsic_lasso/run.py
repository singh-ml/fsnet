#!/usr/bin/env python
import numpy as np
from pyHSICLasso import HSICLasso

hsic_lasso = HSICLasso()
hsic_lasso.input("breast.mat")

hsic_lasso.classification(50)
np.save('features_hl.npy', hsic_lasso.A)
