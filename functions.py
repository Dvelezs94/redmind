import numpy as np


import numpy as np

def cross_entropy_loss(y_pred, y):
    """Cros entropy loss calculation"""
    logprobs = np.multiply(np.log(y_pred), y) + np.multiply((1 - y), np.log(1 - y_pred))
    # loss = -np.mean(logprobs)
    return logprobs

def binary_cross_entropy():
    pass


def mse(y_pred, y):
    return np.mean(np.power(y - y_pred, 2))

def mse_prime(y_pred, y):
    return 2 / (y - y_pred)  / np.size(y)