import numpy as np

def mse(y_val, y_pred):
    return np.mean(np.power(y_val-y_pred, 2))

def mse_prime(y_val, y_pred):
    return 2*(y_pred-y_val)/y_val.size