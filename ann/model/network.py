from typing import Iterator

import numpy as np
from scipy.constants import barrel

from preprocessing.dataloader import DataLoader


# network: List of layers in the network
# loss: Function is used to calculate the loss
# loss_prime: Derivative of loss
# x_train: Input data
# y_train: Output Label
# epoch: Number of training.
# learning_rate
# verbose: If true show


# def train(network, loss, loss_prime, x_train, y_train, epoch=1000, learning_rate=0.01, verbose=True):
#     error_list = []
#     print(f"total images: {len(x_train)}")
#     for e in range(epoch):
#         error = 0
#         for i in range(len(x_train)):
#             print(f"=================epoch {e + 1}, image number {i + 1}====================")
#             x = x_train[i]
#             y = y_train[i]
#
#             # Forward propagation
#             output = x
#             for layer in network:
#                 output = layer.forward(output)
#
#             # Calculate error through the loss function
#             error += loss(y, output)
#
#             print(f"y_true: {[f'{val[0]:.4f}' for val in y]}")
#             print(f"y_pred: {[f'{val[0]:.4f}' for val in output]}")
#
#             # Backward propagation
#             grad = loss_prime(y, output)
#             for layer in reversed(network):
#                 grad = layer.backward(grad, learning_rate)
#
#         error /= len(x_train)
#
#         # Show efficiency
#         if verbose:
#             print(f"Epoch {e + 1}/{epoch}, Error: {error}")
#
#         error_list.append(error)
#
#     print("Errors throughout each epoch:")
#     for err in error_list:
#         print(err)
#
#
# def predict(network, input_data):
#     # zeros(shape, type)
#     results = np.zeros((len(input_data), network[-1].output_size))
#
#     # Consider each input sample
#     for i in range(len(input_data)):
#         output = input_data[i]
#         for layer in network:
#             output = layer.forward(output)
#         results[i] = output
#     return results


def train(network, loss, loss_prime, data_loader: DataLoader, epoch=1000, learning_rate=0.01, verbose=True):
    iterator = iter(data_loader)  # iterator is of type DataLoader.Iterator

    print(f"total images: {data_loader.get_sample_count()}")
    print(f"total batches: {data_loader.get_total_batch()}")
    for e in range(epoch) :
        error = 0
        for batch in iterator:  # batch is of type Batch
            batch_data = batch.batch_data
            batch_labels = batch.batch_labels

            # Forward propagation
            output = predict(network, batch_data)

            error += loss(batch_labels, output)

            # Backward propagation
            grad = loss_prime(batch_labels, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= data_loader.get_total_batch()

        # Show efficiency
        if verbose:
            print(f"Epoch {e + 1}/{epoch}, Error: {error}")


def predict(network, input_batch):
    batch_size = input_batch.shape[0]
    results = np.zeros((batch_size, 4, 1))  # Refer to Label_map in main.py for details

    output = input_batch
    for layer in network:
        output = layer.forward(output)

    results = output

    return results