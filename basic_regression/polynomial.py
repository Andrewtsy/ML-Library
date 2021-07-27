import numpy as np

from .linreg import best_line
from ..preprocessing.basic import scale

def powers_one(x, d):
  X = x.reshape(len(x),1)
  for i in range(1, d):
    X = np.column_stack((X, x * X[:,-1]))
  return X

def powers_two(x, d):
    x_p = x
    for i in range(2, d+1):
        for j in range(i + 1):
            x_p = np.c_[x_p, x[:, 0] ** (i - j) * x[:, 1] ** j]

def best_polynomial(x, y, d, la=0):
    x = powers_one(x, d)
    x_scaled, x_mean, x_std = scale(x)
    y_scaled, y_mean, y_std = scale(y)
    w, b = best_line(x_scaled, y_scaled, la)
    return w, b, x_mean, x_std, y_mean, y_std