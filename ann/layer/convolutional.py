import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer


"""
Y: output
I: input
K: kernel
shape(Y) = shape(I) - shape(K) + 1
"""


class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth):
        # input_shape: width x height x input channels (input channels, height, width)
        # kernel_size: size of kernel (square matrix kernel_size x kernel_size)
        # depth: number of kernels or number of output channels.
        super().__init__()
        self.input_shape = input_shape 
        self.kernel_size = kernel_size
        self.depth = depth                  # input channels
        self.filter = np.random.randn(depth, input_shape[0], kernel_size, kernel_size)
        # kernel_size x kernel_size x input channels x output channels
        self.bias = np.random.randn(depth, 1, 1)
        # number of kernels x 1 x 1
        # each kernel has a unique bias value.

    def forward(self, input):
        # input: width x height x input channels
        # input.shape: (input channels, height, width)
        self.input = input

        # height and width of input image
        shape = np.shape(input)
        in_width = shape[2]
        in_height = shape[1]

        # height and width of output image
        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1
        output = np.zeros((self.depth, out_height, out_width))

        for i in range(self.depth):
            for j in range(self.input_shape[0]):  
                output[i] += signal.correlate(input[j], self.filter[i, j], mode='valid')
            output[i] += self.bias[i]

        return output  

    def backward(self, output_gradient, learning_rate):
        # output_gradient: out_width x out_height x depth (depth, out_height, out_width)
        # update self.bias
        bias_gradient = np.sum(output_gradient, axis=(1, 2), keepdims=True)
        self.bias -= learning_rate * bias_gradient 
        # update self.filter & input_gradient
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_shape[0]):
                self.filter[i, j] -= learning_rate * signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.correlate2d(output_gradient[i], self.filter[i, j], mode='full')
                                            # convolve2d? We have to rotate the filter 180deg
                                            # dE/dX_j = sum(dE/dY_i *full rot(K)) (for i in range [1, input channels]
                                            #         = sum(dE/dY_i *convolve K)
        return input_gradient




