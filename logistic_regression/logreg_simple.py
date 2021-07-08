import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(filename):
    df = pd.read_csv(filename, sep=',')
    arr = np.array(df, dtype=float)
    return arr

def scale(data):
    for i in range(0, data.shape[1]-1):
        data[:,i] = ((data[:,i]) - np.mean(data[:,i])/np.std(data[:,i]))

def plot(x, y):
    plt.xlabel(x[0])
    plt.ylabel(x[1])
    plt.plot(x, y, 'bo')
    plt.show()