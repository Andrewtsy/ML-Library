import numpy as np

from .linreglinalg import best_line
from ..preprocessing.basic import scale

def powers_of_features(x,k):
  X = x.reshape(len(x),1)
  for i in range(1,k):
    X = np.column_stack((X, x*X[:,-1]))
  return X

def best_polynomial(x, y, d, la=0):
    x = powers_of_features(x, d)
    x_scaled, x_mean, x_std = scale(x)
    y_scaled, y_mean, y_std = scale(y)
    w, b = best_line(x_scaled, y_scaled, la)
    return w, b, x_mean, x_std, y_mean, y_std