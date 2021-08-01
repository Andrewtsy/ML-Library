import numpy as np

def layer(W, b, act, layer_type, filter, stride, padding, A):
    return eval(layer_type + '(' + ','.join(filter, stride, A, W, b, act) + ')', \
    {"__builtins__": {}, "dense_layer": fully_connected, "convolutional_layer": convolution, "pooling _layer": pool}, \
    {"W": W, "b": b, "act": act, "layer_type": layer_type, "filter": filter, "stride": stride, "padding": padding, "A": A})
        
