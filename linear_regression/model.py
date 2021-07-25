import numpy as np

from .linreglinalg import best_line
from .polynomial import powers_of_features
from ..preprocessing.basic import shuffle, split_train_test, scale

def linregfull(x, y, prop_train):
    x, y = shuffle(x, y)
    x_train, y_train, x_test, y_test = split_train_test(x, y, prop_train)
    x_train_scaled, x_train_mean, x_train_std = scale(x)
    y_train_scaled, y_train_mean, y_train_std = scale(y)
    w, b = best_line(x_train_scaled, y_train_scaled)
    return w, b, x_train_mean, x_train_std, y_train_mean, y_train_std

def predictions(x, w, b, x_train_mean, x_train_std, y_train_mean, y_train_std):
    x_scaled = (x - x_train_mean) / x_train_std
    y_pred_scaled = np.dot(w, x_scaled) + b
    y_pred = y_train_std * y_pred_scaled + y_train_mean
    return y_pred
    
def polynomial_predictions(x, w, b, x_mean, x_std, y_mean, y_std):
    x = powers_of_features(x, len(w))
    return predictions(x, w, b, x_mean, x_std, y_mean, y_std)