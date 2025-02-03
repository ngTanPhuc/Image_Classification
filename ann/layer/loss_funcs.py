import numpy as np


def mse(y_true, y_pred):
    # TODO
    return np.mean(np.power((y_true - y_pred), 2))


def mse_derivative(y_true, y_pred):
    # TODO: return the vector dE/dY
    return 2 * (y_pred - y_true) / np.size(y_true)


def CCE(y_true, y_pred): # Categorical Cross-entropy
    # TODO
    pass


def CCE_derivative(y_true, y_pred):
    # TODO
    pass
