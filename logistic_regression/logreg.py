import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scales/Normalizes the data
# Scales all columns by default, however, will scale specific columns
# When specififed by iterable of wanted indices
def scale(data, scaled_rows=False):
    if not scaled_rows:
        scaled_rows = range(0, data.shape[1])
    for i in scaled_rows:
        data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:,i]))

# Plots singular feature vs output as specified
def plot(x, y, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'bo')
    plt.show()
    
# Represents hypothesized linear function
def f(x, constants):
    return np.matmul(x, constants)

def cost(x, y, constants):
    return np.matmul((f(x, constants) - y).T, f(x, constants) - y) / (2 * y.shape[0])

# Minimizes the cost/loss function so as to optimize the hypothesized function
def gradient_descent(x, y, constants, learning_rate, num_epochs):
    
    for i in range(num_epochs):
        _cost = np.matmul(x.T, (f(x, constants) - y)) / x.shape[0]
        constants -= learning_rate * _cost
        print(f'loss for epoch {i+1} is {cost(x, y, constants)[0][0]}')
        
    return constants

def model(x, y, constants, learning_rate, epochs):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    y = np.reshape(y, (y.shape[0], 1))
    constants = np.zeros((x.shape[1], 1))
    
    constants = gradient_descent(x, y, constants, learning_rate, epochs)
    
    print('wanted function is f(x)=', end='')
    for i, j in enumerate(constants[1:,0]):
        print(f'{round(j, 3)}x_{i+1}+', end='')
    print(f'{round(constants[0][0], 3)}')
    
    return constants