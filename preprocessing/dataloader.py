import numpy as np
from preprocessing.dataset import *


class DataLoader:
    class Iterator:
        # constructor of class Iterator
        def __init__(self, n, index):
            self.n = n
            self.index = index

        # operator !=
        def __ne__(self, other):
            return self.index != other.index

        # obtain an iterator from an iterable object
        def __iter__(self):
            return self

        # sequential traversal of the elements in the object
        def __next__(self):
            if self.index < len(self.n.batch):
                self.index += 1
                return self.n.batch[self.index-1]
            else:
                raise StopIteration

        # access the elements of an object by index
        def __getitem__(self):
            return self.n.batch[self.index]

    # constructor of class DataLoader
    def __init__(self, arr_dataset, batch_size, shuffle=True, drop_last=False, seed=-1):
        self.arr_dataset = arr_dataset  # array of samples (type TensorDataset)
        self.batch_size = batch_size    # size of each batch
        self.shuffle = shuffle          
        self.drop_last = drop_last      
        self.seed = seed         
        # array self.indices contains integer elements from 0 to the number of samples - 1 (index of each sample)      
        self.indices = np.arange(len(self.arr_dataset))   
        # shuffle arr_dataset if shuffle is true
        if self.shuffle:
            # if seed >=0, shuffling multiple times will yield the same result
            # else the shuffling results will be different each time
            if self.seed >= 0:
                np.random.seed(self.seed)
            # shuffle index of each sample.
            np.random.shuffle(self.indices)
        self.batches = []                                       # array contains result (batches list).
        temp_batch = []                                         # temp array contains each batch of samples
        self.nbatch = len(self.arr_dataset) // self.batch_size  # size of temp_batch (the number of batches)
        # iterate through each batch.
        for i in range(self.nbatch):
            begin = i * self.batch_size
            # at the last batch
            if i == self.nbatch - 1:
                # if drop_last is true, discard the remainder
                if self.drop_last:
                    end = begin + self.batch_size
                # else include the remainder in the last batch
                else:
                    end = len(self.arr_dataset)
            else: 
                end = begin + self.batch_size
            # sort samples include data and label (shuffled if necessary) into each batch.
            indices = self.indices[begin:end]
            temp_data = [] 
            temp_labels = []
            for j in indices: 
                temp_data.append(self.arr_dataset[j].image_data)
                temp_labels.append(self.arr_dataset[j].image_label)  
            data = np.array(temp_data) 
            labels = np.array(temp_labels)
            temp_batch = Batch(data, labels)
            # array contains batches of samples
            self.batches.append(temp_batch)

    # return an Iterator object to iterate over the elements in the batch
    def __iter__(self):
        return self.Iterator(self, 0)

    # return the element at the index in the batch
    def __getitem__(self, index):
        if index < len(self.batches):
            return self.batches[index]
        else:
            raise IndexError("Index out of range")

    # get number of samples in each batch  
    def get_batch_size(self):
        return self.batch_size

    # get number of samples
    def get_sample_count(self):
        return len(self.arr_dataset)

    # get number of batches
    def get_total_batch(self):
        return self.nbatch

