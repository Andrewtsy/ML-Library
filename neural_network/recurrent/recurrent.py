import numpy as np
from ..function.activations import sigmoid

class RNN():
    def __init__():
        pass
    
    def SRNN(self, inp_shape):    
        self.n_t = inp_shape[0]
        self.n_h = inp_shape[1]
    
        self.W_h = np.random.random((self.n_x, self.n_h))
        self.b_x = np.random.random((self.n_h, self.n_h))
        self.b_h = np.random.random((self.n_h))
    
    def SRNN_forward_prop(self, x):
        self.n_e = x.shape[0]
        H = np.tanh(np.dot(self.n_e, self.W_x) + self.b_h).reshape(-1, 1, self.n_h)
        for i in range(1, x.shape[0]):
            H = np.append(H, np.tanh(np.dot(H[i-1], self.W_h) + np.dot(x[i], self.W_x) + self.b_h).reshape(-1, 1, self.n_h), axis=0)
        return H
    
    def LSTM(self, inp_shape):
        self.n_x, self.t, self.n_h = *inp_shape
        self.W_xf = np.random.random((self.n_x, self.n_h))
        self.W_hf = np.random.random((self.n_h, self.n_h))
        self.b_f = np.random.random(self.n_h)
        
        self.W_xi = np.random.random((self.n_x, self.n_h))
        self.W_hi = np.random.random((self.n_h, self.n_h))
        self.b_i = np.random.random(self.n_h)
        
        self.W_xc = np.random.random((self.n_x, self.n_h))
        self.W_hc = np.random.random((self.n_h, self.n_h))
        self.b_c = np.random.random(self.n_h)
        
        self.W_xo = np.random.random((self.n_x, self.n_h))
        self.W_ho = np.random.random((self.n_h, self.n_h))
        self.b_o = np.random.random(self.n_h)
    
    def LSTM_forward_prop(self, x):
        self.n_e = x
        self.Y = np.zeros((self.n_e, self.n_t, self.n_h))
        self.h = np.zeros((self.n_e, self.n_h))
        self.c = np.zeros((self.n_e, self.n_h))
        for j in range(self.n_t):
            self.f = sigmoid(np.dot(x[:, j, :], self.W_xf) + np.dot(self.h, self.W_hf) + self.b_f)
            self.i = sigmoid(np.dot(x[:, j, :], self.W_xi) + np.dot(self.h, self.W_hi) + self.b_i)
            self.c_tilde = np.tanh(np.dot(x[:, j, :], self.W_xc) + np.dot(self.h, self.W_ho) + self.b_o)
            self.o = sigmoid(np.dot(x[:, j, :], self.W_xo) + np.dot(self.h, self.W_ho) + self.b_o)
            self.c = self.i * self.c_tilde + self.f * self.c
            self.h = self.o * np.tanh(self.c)
            self.Y[:, j, :] = self.h
        return self.Y
    
    def GRU(self, inp_shape):
        self.n_x, self.n_h, self.t = *inp_shape
        self.W_xr = np.random.random((self.n_x, self.n_h))
        self.W_hr = np.random.random((self.n_h, self.n_h))
        self.b_r = np.random.random(self.n_h)
        
        self.W_xz = np.random.random((self.n_x, self.n_h))
        self.W_hz = np.random.random((self.n_h, self.n_h))
        self.b_z = np.random.random(self.n_h)
        
        self.W_xo = np.random.random((self.n_x, self.n_h))
        self.W_ho = np.random.random((self.n_h, self.n_h))
        self.b_o = np.random.random(self.n_h)
        
    def GRU_forward_prop(self, x):
        self.n_e = x.shape[0]
        self.Y = np.zeros((self.n_e, self.n_t, self.n_h))
        self.h = np.zeros((self.n_e, self.n_h))
        self.c = np.zeros((self.n_e, self.n_h))
        for j in range(self.n_t):
            self.f = sigmoid(np.dot(x[:, j, :], self.W_xf) + np.dot(self.h, self.W_hf) + self.b_f)
            self.i = sigmoid(np.dot(x[:, j, :], self.W_xi) + np.dot(self.h, self.W_hi) + self.b_i)
            self.c_tilde = np.tanh(np.dot(x[:, j, :], self.W_xc) + np.dot(self.h, self.W_ho) + self.b_o)
            self.o = sigmoid(np.dot(x[:, j, :], self.W_xo) + np.dot(self.h, self.W_ho) + self.b_o)
            self.c = self.i * self.c_tilde + self.f * self.c
            self.h = self.o * np.tanh(self.c)
            self.Y[:, j, :] = self.h
        return self.Y