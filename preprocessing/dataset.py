import numpy as np
from abc import ABC, abstractmethod


# e.g. The image's size  is 28x28 (a 2D matrix). So the data variable will be a 2D matrix.
# If each batch contains 10 images, then the data variable in it is 10x28x28 (a 3D matrix)

# The label variable in the DataLabel class is a 1D matrix of size (1x4) (horse, cat, dog, chicken)
# The element at the corresponding position to the animal will be 1 and the rest will be 0

class DataLabel:  # This class will store a pair of 1 image and its label
    def __init__(self, image_data, image_label):
        # TODO
        self.image_data = image_data
        self.image_label = image_label


class Batch:  # This class will store multiple images
    def __init__(self, batch_data, batch_labels):
        # TODO
        self.batch_data = batch_data
        self.batch_labels = batch_labels


class Dataset(ABC):
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def getimage(self, index):  # This is the getitem method in the major assignment 1
        pass


# This class and its data variable will hold the whole dataset
# All the images will be load into this class in the implementation step
# e.g.
# images = np.array([image0, image1,image2])
# labels = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1])
# TensorDataset data_images(images, labels)
class TensorDataset(Dataset):
    def __init__(self, dataset, labels):
        # TODO: need to initialize the attributes:
        #   * 1. data, label;
        #   * 2. data_shape, label_shape
        self.dataset = dataset
        self.labels = labels

    def length(self):
        # TODO: return the size of the first dimension (dimension 0)
        return self.dataset.shape[0]

    def getimage(self, index) -> DataLabel:  # This is the getitem method in the major assignment 1
        # TODO: return the data item (of type: DataLabel) that is specified by index
        if index < 0 or index > self.length():
            raise IndexError("Invalid index!")

        return DataLabel(self.dataset[index], self.labels[index])
