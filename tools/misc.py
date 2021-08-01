import numpy as np

def norm(W):
    return np.sum([W[i] ** 2 for i in range(len(W))])