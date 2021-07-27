import numpy as np

def scale(x):
    mu = np.mean(x)
    st = np.std(x)
    return (x-mu) / st, mu, st

def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def split_train_test(x, y, prop_train):
    n = int(prop_train * len(x))
    return x[:n], y[:n], x[n:], y[n:]

def split_train_test_val(x, y, prop_train, prop_test):
    n = int(prop_train * len(x))
    m = int(prop_test * len(x)) + n
    return x[:n], y[:n], x[n:m], y[n:m], x[m:], y[m:]