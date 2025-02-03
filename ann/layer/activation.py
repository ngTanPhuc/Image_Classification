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
from ILayer import ILayer


class Activation(ILayer):
    def __init__(self, activation, activation_derivative):
        # TODO
        super().__init__()

    def forward(self, X):
        # TODO
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO
        pass
