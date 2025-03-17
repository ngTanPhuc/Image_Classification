import numpy as np
from ann.layer.ILayer import ILayer


class Reshape(ILayer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        self.batch_size = np.shape(X)[0]
        return np.reshape(X, (self.batch_size, self.output_shape[0], self.output_shape[1]))

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)