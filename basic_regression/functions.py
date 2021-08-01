import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def bce(y, y_pred):
    -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / len(y)

def bce_gradient_w(x, y, y_pred, w, la):
    return np.matmul(y_pred - y, x) / len(y) + 2 * la * w / len(y)

def bce_gradient_b(x, y, y_pred):
    return np.sum(y_pred - y) / len(y)

def mse(y_val, y_pred):
    return np.mean(np.power(y_val-y_pred, 2))

def mse_gradient(y, y_pred, w, la):
    return 2 * (y_pred-y) / y.size + 2 * la * w / len(y)