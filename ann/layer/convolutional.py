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
    def __init__(self, input_image_shape, kernel_size, depth):
        # input_image_shape: width x height x input channels (input channels, height, width)
        # kernel_size: size of kernel (square matrix kernel_size x kernel_size)
        # depth: number of kernels or number of output channels.
        super().__init__()
        self.input_image_shape = input_image_shape
        self.kernel_size = kernel_size
        self.depth = depth                   # input channels
        self.filter = np.random.randn(depth, input_image_shape[0], kernel_size, kernel_size)
        # kernel_size x kernel_size x input channels x output channels

                                                # input height
        self.bias = np.random.randn(depth, self.input_image_shape[1] - self.kernel_size + 1,
                                             # input width
                                        self.input_image_shape[2] - self.kernel_size + 1)
        # bias_size = output_size = (depth, input height - kernel_size + 1, input width - kernel size + 1)
        # each kernel has a unique bias matrix

    def forward(self, X):
        # input: width x height x input channels x batch_size
        # input.shape: (batch_size, input channels, height, width)
        self.input = X

        # height and width of an input image
        shape = np.shape(self.input)
        self.batch_size = shape[0]
        in_height = shape[2]
        in_width = shape[3]

        # height and width of output image
        out_height = in_height - self.kernel_size + 1
        out_width = in_width - self.kernel_size + 1
        output = np.zeros((self.batch_size, self.depth, out_height, out_width))

        for data_idx in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_image_shape[0]):
                    output[data_idx, i] += signal.correlate(self.input[data_idx, j], self.filter[i, j], mode='valid')
                output[data_idx][i] += self.bias[i]

        return output  

    def backward(self, output_gradient, learning_rate):
        # output_gradient: out_width x out_height x depth (batch_size, depth, out_height, out_width)
        # update self.bias
        bias_gradient = np.mean(output_gradient, axis=0, keepdims=True)  # (1, depth, output_height, output_width)
        bias_gradient = np.squeeze(bias_gradient, axis=0)  # (depth, output_height, output_width)
        self.bias -= learning_rate * bias_gradient

        # height and width of an input image
        shape = np.shape(self.input)
        self.batch_size = shape[0]

        # update self.filter & input_gradient
        input_gradient = np.zeros((self.batch_size, ) + self.input_image_shape)
        filter_gradient = np.zeros_like(self.filter)
        for data_idx in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_image_shape[0]):
                    filter_gradient[i, j] += signal.correlate2d(self.input[data_idx, j],
                                                                output_gradient[data_idx, i], mode='valid')

                    input_gradient[data_idx, j] += signal.convolve2d(output_gradient[data_idx, i],
                                                                     np.rot90(self.filter[i, j], 2) , mode='full')

        filter_gradient /= self.batch_size
        self.filter -= learning_rate * filter_gradient

        return input_gradient




