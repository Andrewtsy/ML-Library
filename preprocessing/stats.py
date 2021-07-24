import numpy as np

def scale(x):
    mu = np.mean(x)
    st = np.std(x)
    return (x-mu) / st, mu, st