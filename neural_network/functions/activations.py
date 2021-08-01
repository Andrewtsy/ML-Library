import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))
    
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1 * (x >= 0)

def identity(x):
    return x

def identity_derivative(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return np.cosh(x)

def arctan(x):
    return np.arctan(x)

def arctan_derivative(x):
    return 1 / (1 + x ** 2)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1, 1))

def temp(x):
    pass

def h(x, act='None'):
    return eval(act + '(x)', {"__builtins__": {}, "sigmoid": sigmoid, "relu": relu, "identity": identity, "tanh": tanh, "softmax": softmax, "None": temp}, {"act": act, "x": x})

def h_derivative(x, act):
    return eval(act + '(x)', {"__builtins__": {}, "sigmoid": sigmoid_derivative, "relu": relu_derivative, "identity": identity_derivative, "tanh": tanh_derivative, "softmax": temp}, {"act": act, "x": x})