# Implement the activation functions needed for the neural network (e.g. Tanh, Sigmoid, ReLU,...)
import numpy as np
from ILayer import ILayer
from activation import Activation


class Softmax(ILayer):
    def __init__(self):
        # TODO
        pass

    def forward(self, input):
        # TODO
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO
        pass


class Sigmoid(Activation):
    def __init__(self):
        # TODO
        def sigmoid():
            # TODO
            pass

        def sigmoid_derivative():
            # TODO
            pass


class ReLU(Activation):
    def __init__(self):
        # TODO
        def relu():
            # TODO
            pass

        def relu_derivative():
            # TODO
            pass


class LeakyReLU(Activation):
    def __init__(self):
        # TODO
        def leaky_relu():
            # TODO
            pass

        def leaky_relu_derivative():
            # TODO
            pass


class Tanh(Activation):
    def __init__(self):
        # TODO
        def tanh():
            # TODO
            pass

        def tanh_derivative():
            # TODO
            pass
