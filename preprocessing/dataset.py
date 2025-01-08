import numpy as np
from abc import ABC, abstractmethod

class DataLabel:
    def __init__(self, data, label):
        # TODO
        pass


class Batch:
    def __init__(self, data, label):
        # TODO
        pass


class Dataset(ABC):
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def getimage(self, index):  # This is the getitem method in the major assignment 1
        pass


class TensorDataset(Dataset):
    def __init__(self, data, label):
        # TODO: need to initialize the attributes:
        #   * 1. data, label;
        #   * 2. data_shape, label_shape
        pass

    def length(self):
        # TODO: return the size of the first dimension (dimension 0)
        pass

    def getimage(self, index):  # This is the getitem method in the major assignment 1
        # TODO: return the data item (of type: DataLabel) that is specified by index
        pass
