import numpy as np
from .temp_layer import Layer

# Dense or Fully Connected Layer (affectionately nicknamed 'Vanilla')        
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # populates layer with random parameters
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    def forward_propagation(self, input_data):
        # weighted sum of inputs and biases through feedforward network (prediction)
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        # calculation of sum of errors ie. sum of partial derivative of loss according to individual weights (training)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        # adjustment of weights/biases
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error