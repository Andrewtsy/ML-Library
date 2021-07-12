import numpy as np

from functions.activations import tanh, tanh_prime
from functions.losses import mse, mse_prime
from layers.vanilla_layer import Dense
from layers.activation import ActivationLayer
from models.models import Network

# Sample script running the neural network on an AND gate task

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[0]], [[0]], [[1]]])

net = Network()
net.add(Dense(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dense(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

net.fit(x_train, y_train, epochs=500, learning_rate=0.1)
print(net.predict(x_train))