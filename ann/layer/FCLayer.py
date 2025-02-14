"""
    w_ij
x_i -----> y_j = w_1j.x_1 + w_2j.x_2 +...+ w_ij.x_i + b_j


    y_1        w_11  w_12  ...  w_1i         x_1           b_1
    y_2        w_21  w_22  ...  w_2i         x_2           b_2
    .     =   .    .    .    .     x     .      +     .
    .         .    .     .   .           .            .
    .         .    .      .  .           .            .
    y_j        w_j1  w_j2  ...  w_ji         x_i           b_j

    Y = W x X + B

Y(j x 1)
W(j x i)
X(i x 1)
B(j x 1)
"""


import numpy as np
from ann.layer.ILayer import ILayer


class FCLayer(ILayer):
    def __init__(self, X_size, Y_size):
        super().__init__()
        rgn = np.random.default_rng()  # random number generator
        self.Weight = rgn.standard_normal(size=(Y_size, X_size)) * 0.001  # Small initial weights
        self.Bias = rgn.random(size=(Y_size, 1))

    def forward(self, X):
        # TODO: return output Y
        self.input = X  # Store the input for the backward propagation
        self.output = np.dot(self.Weight, self.input) + self.Bias
        return self.output

    def backward(self, output_gradient, learning_rate):
                        # dE/dY(j x 1)
        # TODO: update Weight and Bias, then return input gradient X_gradient
        """
        from the given dE/dY, we need to compute dE/dX, dE/dW and dE/dB
        +Compute dE/dW:
            dE/dW = dE/dY . dY/dW
            given that: y_j = w_1j.x_1 + w_2j.x_2 +...+ w_ij.x_i + b_j
            we can see that dy_j/dw_m = x_m (for m in range [1, i])
            ==> therefore, dE/dW = dE/dY . X^T
        +Compute dE/dB:
            dE/dB = dE/dY . dY/dB
            given that: y_j = w_1j.x_1 + w_2j.x_2 +...+ w_ij.x_i + b_j
            we can see that dy_j/db_j = 1
            ==> therefore, dE/dB = dE/dY
        +Compute dE/dX:
            dE/dX = dE/dY . dY/dX
            given that: y_j = w_1j.x_1 + w_2j.x_2 +...+ w_ij.x_i + b_j
            we have dE/dx_m = dE/dy_1.dy_1/dx_m + dE/dy_2.dy_2/dx_m +...+ dE/dy_j.dy_j/dx_m (for m in range [1, i])
                            = dE/dy_1.w_1m + dE/dy_2.w_2m +...+ dE/dy_j.w_jm
            ==> therefore, dE/dX = W^T . dE/dY
        """
        Weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        Bias_gradient = output_gradient
        self.Weight -= learning_rate * Weight_gradient
        self.Bias -= learning_rate * Bias_gradient
        X_gradient = np.dot(np.transpose(self.Weight), output_gradient)
        return X_gradient
