"""This script allows for multiple features and uses linear algebra
It takes as input numpy arrays of the features and labels."""

import numpy as np

def add_one_end_rows(x):
    return np.column_stack((x, np.ones(x.shape[0])))

def best_line(x, y, la=0):
    """Returns parameters for the linear regression model"""
    C = add_one_end_rows(x)

    H = np.eye(x.shape[1] + 1)
    H[-1, -1] = 0

    A = np.matmul(C.T, C) + la * H
    r = np.dot(C.T, y)
    z = np.linalg.solve(A, r)
    w = z[:-1]
    b = z[-1]
    return w, b

def f(x, w, b):
    """Returns predictions based on parameters"""
    return np.dot(w, x) + b