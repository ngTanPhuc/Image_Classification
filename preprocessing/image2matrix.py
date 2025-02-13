# This is the 1st step which changes the images into matrices of integers (grayscale or RGB)
# This file contains a function which return the matrix of an input image. The function will then
# be called each time we fetch an image into the dataset of the TensorDataset

# Grayscale: 2D matrix/tensor with each element represent the brightness (from 0 to 255 or from black to white)
#   pros: smaller in size, reduce the amount of data to make the training faster, useful when color is not important
#   cons: loss of information
# RGB: 3D matrix/tensor with the first 2 dimensions represent the width and height of the image; whereas the third
# dimension represent the color channel (R, G, B) which stands for red, green and blue
# You may add additional libraries to help you with this
import numpy as np
from PIL import Image


def img2matr(file_path: str, img_size=(64, 64)):
    image = Image.open(file_path)  # Open the image
    image = image.convert("RGB")  # Convert the image to RGB
    image = image.resize(img_size)

    matrix = np.array(image)  # Convert the image into matrix (height, width, channels)

    # Convert to (channels, height, width) to pass to Convolutional layer
    input_image = np.transpose(matrix, (2, 0, 1))
    input_image = input_image.astype(np.float32)  # Convert the data type from uint8 to float32 to use for division
    input_image /= 255.0

    return input_image
