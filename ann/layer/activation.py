"""
The activation function f will calculate y* to determine whether the neuron should be activated (on/off) or not
y*_1 = f(y_1)
y*_2 = f(y_2)
.
.
.
y*_j = f(y_j)

         X  --->  +------------------+  ------->   Y   ----->   +------------------+  ---> Y*
                  |      Layer       |                          | Activation Layer |
                  |        W         |                          |                  |
     dE/dX  <---  +------------------+  <-----  dE/dY  <-----   +------------------+  <--- dE/dY*
                          |
                          v
                        dE/dW
"""

import numpy as np
from ann.layer.ILayer import ILayer


class Activation(ILayer):
    def __init__(self, activation, activation_derivative):
        # TODO
        # activation and activation_derivative are the functions of the Activation layer
        super().__init__()
        self.activation = activation
        self. activation_derivative = activation_derivative

    def forward(self, X):
        # TODO: return the result of the activation function
        self.input = X
        input_shape = np.shape(self.input)
        self.batch_size = input_shape[0]
        self.output = np.zeros(input_shape)

        for data_idx in range(self.batch_size):
            self.output[data_idx] = self.activation(self.input[data_idx])

        return self.output

    def backward(self, output_gradient, learning_rate):
                        # dE/dY*(j x 1)
        # TODO: calculate and return the dE/dY
        """
        from the given dE/dY*, we need to calculate the dE/dY
        we have dE/dy_1 = dE/y*_1 . dy*_1/dy_1 + dE/y*_2 . dy*_1/dy_1 +...+ dE/y*_i . dy*_1/dy_1
        we can see that dE/dy_1 = dE/y*_1 . dy*_1/dy_1
                                = dE/y*1 . f'(y_1)
        therefore, dE/dY = dE/dY* ⊙ f'(Y)
        """
        return np.multiply(output_gradient, self.activation_derivative(self.input))
