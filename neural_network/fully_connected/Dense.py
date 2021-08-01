import numpy as np

from ..functions.activations import h, h_derivative
from ..functions.errors import error


def forward_prop(X, W, b, act):
    """Inputs:
    act[l] : 'sigmoid, 'relu', 'identity' or 'hyperbolic tangent' or the activation function in layer l
    W : a list with the weights, with each w[l] being  a numpy array for layer l
    b : a list with the biases, with each b[l] being  a numpy array for layer l
    X : matrix of features of examples
    Outputs:
    list of As and Zs
    """
    A = [X]
    Z = [0]
    L = len(b) - 1
    for l in range(1, L+1):
        Z.append(np.matmul(A[-1], W[l]) + b[l])
        A.append(h(Z[-1], act[l]))
    return A, Z

def initialize(n):
    W = [0]
    b = [0]
    for l in range(1, len(n)):
        W.append(np.random.randn(n[l-1], n[l]) / np.sqrt(n[l-1]))
        b.append(np.zeros(n[l]))
    return W, b

def gradients(A, Z, act, W, y):
    """Input
    n[l] :  number of nodes in layer l
    act[l] : 'sigmoid, 'relu', 'identity' or 'hyperbolic tangent' or the activation function in layer l
    W : a list with the weights, with each w[l] being  a numpy array for layer l
    b : a list with the biases, with each b[l] being  a numpy array for layer l
    X : matrix of features of examples
    Output
    Gradients of the error with respect to the weights and biases DJ_DW and DJ_Db
    """
    L = len(W) - 1
    y_pred = A[-1]
    DJ_DW = []
    DJ_Db = []
    DJ_DZ = (y_pred - y.reshape(-1, 1)) / len(y)
    for l in reversed(range(1, L + 1)):
        DJ_DA = np.matmul(DJ_DZ, W[l].T)
        DJ_DW.insert(0, np.matmul(A[l-1].T, DJ_DZ))
        DJ_Db.insert(0, np.sum(DJ_DZ, axis=0))
        DJ_DZ = DJ_DA * h_derivative(Z[l-1], act[l-1])
    DJ_DW.insert(0, 0)
    DJ_Db.insert(0, 0)
    return DJ_DW, DJ_Db

def update(W, b, DJ_DW, DJ_Db, c):
    for l in range(1, len(b)):
        W[l] = W[l] - c * DJ_DW[l]
        b[l] = b[l] - c * DJ_Db[l]
    return W, b

def update_reg(W, b, DJ_DW, DJ_Db, c, la=0):
    for l in range(len(b)):
        W[l] = W[l] - c * DJ_DW[l] - 2 * c * la * W[l] / len(b)
        b[l] = b[l] - c * DJ_Db[l]
    return W, b

def gradient_descent(n, act, X, y, epochs, c):
    W, b = initialize(n)
    J_list = []
    for i in range(epochs):
        A, Z = forward_prop(X, W, b, act)
        y_pred = A[-1]
        J_list.append(error(y, y_pred.reshape(-1,1)))
        DJ_DW, DJ_Db = gradients(A, Z, act, W, y)
        W, b = update(W, b, DJ_DW, DJ_Db, c)
    return W, b, J_list

def gradient_descent_reg(n, act, X, y, epochs, c, la=0):
    W, b = initialize(n)
    J_list = []
    for i in range(epochs):
        A, Z = forward_prop(X, W, b, act)
        DJ_DW, DJ_Db = gradients(A, Z, act, W, y)
        W, b = update_reg(W, b, DJ_DW, DJ_Db, c, la)
        J_list.append(error(y, A[-1].reshape(-1,1)))
    return W, b, J_list

def predict(X, W, b, act):
    A, Z = forward_prop(X, W, b, act)
    return A[-1].reshape(-1)

def affine(X, W, b):
    return np.matmul(X, W) + b

def fully_connected(X, W, b, act):
    Z = affine(X, W, b)
    A = h(act, Z)
    return A, Z