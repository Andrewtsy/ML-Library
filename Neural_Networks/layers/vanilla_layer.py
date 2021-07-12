import numpy as np
from .temp_layer import Layer

# FC/Dense layer
class Dense(Layer):
    """Dense or Fully Connected Layer (affectionately nicknamed 'Vanilla') cuz ya basic"""
    def __init__(self, input_size, output_size):
        # Populates/initializes layer with random parameters
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    def forward_propagation(self, input_data):
        # Weighted sum of inputs and biases through feedforward network (prediction)
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        # Calculation of sum of errors ie. sum of partial derivative of loss according to individual weights (training)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        # Adjustment of weights/biases
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error