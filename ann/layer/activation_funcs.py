# Implement the activation functions needed for the neural network (e.g. Tanh, Sigmoid, ReLU,...)
import numpy as np
from ann.layer.ILayer import ILayer
from ann.layer.activation import Activation


class Softmax(ILayer):
    def forward(self, input):
        # TODO
        self.input = input
        # Subtract the number with the largest number to prevent the exp too large
        temp = np.exp(self.input - np.max(self.input))
        self.output = temp / np.sum(temp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # TODO
        temp = self.output.reshape(-1, 1)  # Convert to column vector
        jacobian = np.diagflat(temp) - np.dot(temp, np.transpose(temp))

        return np.dot(jacobian, output_gradient)


class Sigmoid(Activation):
    def __init__(self):
        # TODO
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            # TODO
            """
            let u = 1 - np.exp(-x)
            and f'(x) = -np.exp(-x) / [(1 - np.exp(-x)) ** 2]
            ==> f'(x) = (u - 1) / u^2 = (1/u) * [1 - (1/u)]
            but 1/u = sigmoid(x)
            therefore,
            """
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        # TODO
        def relu(x):
            # TODO
            return max(0, x)

        def relu_derivative(x):
            # TODO
            return 1 if x > 0 else 0

        super().__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        # TODO
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 / (np.cosh(x) ** 2)

        super().__init__(tanh, tanh_derivative)
