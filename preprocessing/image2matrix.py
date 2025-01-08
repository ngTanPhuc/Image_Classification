# This is the 1st step which change the images into matrices of integers (grayscale or RGB)
# This file contains a function which return the matrix of an input image. The function will then
# be called in the dataset.py

# Grayscale: 2D matrix/tensor with each element represent the brightness (from 0 to 255 or from black to white)
#   pros: smaller in size, reduce the amount of data to make the training faster, useful when color is not important
#   cons: loss of information
# RGB: 3D matrix/tensor with the first 2 dimensions represent the width and height of the image; whereas the third
# dimension represent the color channel (R, G, B) which stands for red, green and blue
# You may add additional libraries to help you with this
import numpy as np