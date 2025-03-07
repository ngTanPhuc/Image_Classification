import numpy as np


# network: List of layers in the network
# loss: Function is used to calculate the loss
# loss_prime: Derivative of loss
# x_train: Input data
# y_train: Output Label
# epoch: Number of training.
# learning_rate
# verbose: If true show


def train(network, loss, loss_prime, x_train, y_train, epoch=1000, learning_rate=0.01, verbose=True):
    error_list = []
    print(f"total images: {len(x_train)}")
    for e in range(epoch):
        error = 0
        for i in range(len(x_train)):
            print(f"=================epoch {e + 1}, image number {i + 1}====================")
            x = x_train[i]
            y = y_train[i]

            # Forward propagation
            output = x
            for layer in network:
                output = layer.forward(output)

            # Calculate error through the loss function
            error += loss(y, output)

            print(f"y_true: {[f'{val[0]:.4f}' for val in y]}")
            print(f"y_pred: {[f'{val[0]:.4f}' for val in output]}")

            # Backward propagation
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)

        # Show efficiency
        if verbose:
            print(f"Epoch {e + 1}/{epoch}, Error: {error}")

        error_list.append(error)

    print("Errors throughout each epoch:")
    for err in error_list:
        print(err)


def predict(network, input_data):
    # zeros(shape, type)
    results = np.zeros((len(input_data), network[-1].output_size))

    # Consider each input sample
    for i in range(len(input_data)):
        output = input_data[i]
        for layer in network:
            output = layer.forward(output)
        results[i] = output
    return results
