import numpy as np

def mse(y, y_pred):
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

def mse_prime(y, y_pred):
    return 2 * (y - y_pred)