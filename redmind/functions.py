import numpy as np

########################
# Regression functions #
########################
def mse(y, y_pred) -> np.float64:
    squared_error = np.power(y_pred - y, 2)
    return np.mean(squared_error)
    
def mse_prime(y, y_pred) -> np.ndarray:
    return 2 * (y_pred - y)

#########################
# Categorical functions #
#########################

##
# Multi class classification [0 0 1 0 0]
# or one hot encoded vector
##
def cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred))

def cross_entropy_prime(y, y_pred):
    return y_pred - y

##
# two-class / binary classification  [0 1] [0] [1]
##
def binary_cross_entropy(y ,y_pred) -> np.float64:
    return  -np.mean((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

def binary_cross_entropy_prime(y, y_pred) -> np.ndarray:
    return -((y_pred-y) / ((y_pred - 1) * y_pred))

#################################
# Learning rate decay functions #
#################################
def lr_decay(learning_rate, epoch, decay_rate):
    """Standard learning rate decay algorithm"""
    return (1 / (1 + decay_rate * epoch)) * learning_rate