import numpy as np

# Each batch contains the nonlinear (except for id) activation function and its derivative
# Former used for forward prop, latter for back prop

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.power(np.tanh(x), 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.array([i if i > 0 else 0 for i in x])

def relu_prime(x):
    return np.array([1 if i > 0 else 0 for i in x])  


def identity(x):
    return x

def identity_prime(x):
    return np.ones(x.shape)