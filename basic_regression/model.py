import numpy as np

from .simple_linreg_2 import best_line
from .functions import mse, mse_gradient, sigmoid, \
bce, bce_gradient_b, bce_gradient_w
from ..preprocessing.basic import shuffle, split_train_test, scale


class linear_regression:
    def __init__(self):
        pass
    
    def fit_gd(self, x, y, r, epochs, la=0):
        k = x.shape[1]
        self.w = np.random.randn(k)
        self.b = 0
        self.w_list = np.empty((0, k), float)
        self.b_list = np.empty(0, float)
        self.J_list = np.empty(0, float)
        for i in range(epochs):
            y_pred = self.predictions(x, self.w, self.b)
            self.w -= r * mse_gradient(y, y_pred, self.w, la)
            self.b -= r * mse_gradient(y, y_pred, self.w, la)
            self.w_list = np.append(self.w_list, [self.w], axis=0)
            self.b_list = np.append(self.b_list, [self.b], axis=0)
            J = mse(y, y_pred)
            self.J_list = np.append(self.J_list, [J])
        return self.w, self.b, self.J_list
    
    def fit_la(self, x, y, la=0):
        self.w, self.b = best_line(x, y, la)

    def predictions(self, x):
        y_pred = np.dot(self.w, x) + self.b
        return y_pred

    def scaler_predictions(self, x, x_train_mean, x_train_std, y_train_mean, y_train_std):
        x_scaled = (x - x_train_mean) / x_train_std
        y_pred_scaled = self.predictions(x_scaled)
        y_pred = y_train_std * y_pred_scaled + y_train_mean
        return y_pred


class logistic_regression:
    def __init__(self):
        pass
    
    def fit_gd(self, x, y, r, epochs, la=0):
        k = x.shape[1]
        self.w = 0.1 * np.random.randn(k)
        self.b = 0
        self.w_list = np.empty((0, k), float)
        self.b_list = np.empty(0, float)
        self.J_list = np.empty(0, float)
        for i in range(epochs):
            y_pred = self.predictions(x, self.w, self.b)
            self.w -= r * bce_gradient_w(x, y, y_pred, self.w, la)
            self.b -= r * bce_gradient_b(x, y, y_pred, self.w, la)
            self.w_list = np.append(self.w_list, [self.w], axis=0)
            self.b_list = np.append(self.b_list, [self.b], axis=0)
            J = bce(y, y_pred)
            self.J_list = np.append(self.J_list, [J])
        return self.w, self.b, self.J_list

    def predictions(self, x):
        y_pred = sigmoid(np.dot(self.w, x) + self.b)
        return y_pred

    def scaler_predictions(self, x, x_train_mean, x_train_std, y_train_mean, y_train_std):
        x_scaled = (x - x_train_mean) / x_train_std
        y_pred_scaled = self.predictions(x_scaled)
        y_pred = y_train_std * y_pred_scaled + y_train_mean
        return y_pred


def linreg_full_sample(self, x, y, prop_train):
        x, y = shuffle(x, y)
        x_train, y_train, x_test, y_test = split_train_test(x, y, prop_train)
        x_train_scaled, x_train_mean, x_train_std = scale(x)
        y_train_scaled, y_train_mean, y_train_std = scale(y)
        w, b = best_line(x_train_scaled, y_train_scaled)
        return w, b, x_train_mean, x_train_std, y_train_mean, y_train_std