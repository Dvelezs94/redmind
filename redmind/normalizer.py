import numpy as np
class Normalizer():
    """
    Class to normalize input features
    """
    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.axis = 0

    def fit(self, X: np.ndarray, axis=0) -> None:
        """
        Set normalizer mean and variance to fit the data.
        """
        self.axis = axis
        self.mean = np.mean(X, axis=self.axis)
        self.std = np.std(X - self.mean, axis=self.axis)

    def scale(self, X: np.ndarray):
        """
        Returns normalized input
        """
        return (X - self.mean) / self.std

