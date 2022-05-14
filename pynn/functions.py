import numpy as np

def mse(y_pred, y):
    """
    Calculates difference between predicted value and actual value
    and squares them (Mean Squared Error)
    Returns a 
    
    Inputs
    ------
    y_pred
    y

    Outputs
    -------
    squared_errors: 
    """
    return np.mean(np.power(y - y_pred, 2))

def mse_prime(y_pred, y):
    return 2 * (y_pred - y) / np.size(y)