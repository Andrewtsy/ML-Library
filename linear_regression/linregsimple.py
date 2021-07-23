"""Small, basic functions for single feature linear regression.
This assumes data is roughly linear with one feature and may be modeled by:
y = wx + b
All calculations calculated with system of equations/quadratics (no gradient descent)
"""


import math

import numpy as np
import matplotlib.pyplot as plt

def lineqsolver(a_11, a_12, b_1, a_21, a_22, b_2):
    """Returns solution of a system of equations of the form:
    a_11x_1 + a_12x_2 = b_1
    a_21x_1 + a_22x_2 = b_2
    """
    d = a_11 * a_22 - a_21 * a_21
    x_1 = (a_22 * b_1 - a_12 * b_2) /d
    x_2 = (-a_21 * b_1 + a_11 * b_2) / d

    return x_1, x_2

def lineqmakersimp(x, y):
    """Returns systems of equations that represents the solutions to
    w and b in minimizing the MSE loss function.
    This function sovery naive to the point that it uses lists of data points,
    not numpy arrays (just for fun)
    """
    a_11 = sum(map(lambda i: i**2, x))
    a_12 = sum(x)
    a_21 = a_12
    a_22 = len(x)
    b_1 = sum(map(lambda i: i[0] * i[1], (zip(x, y))))
    b_2 = sum(y)

    return a_11, a_12, b_1, a_21, a_22, b_2

def lineqmaker(x, y):
    """Returns systems of equations that represents the solutions to
    w and b in minimizing the MSE loss function.
    This function does use numpy arrayss
    """
    a_11 = (x*x).sum()
    a_12 = x.sum()
    b_1 = (x*y).sum()
    a_21 = a_12
    a_22 = len(x)
    b_2 = y.sum()

    return a_11, a_12, b_1, a_21, a_22, b_2

def linreg(x, y):
    """Returns parameters w and b for the linear regression model"""
    return lineqsolver(*lineqmaker(x, y))

def f(w, b, x):
    """Applies the linear regression model to predict y values for
    an iterable of x values""""
    return [w * i + b for in x]