import numpy as np
class Normalizer():
    """
    Class to normalize input features
    """
    def __init__(self):
        self.mean = 0
        self.variance = 0

    def fit(self, X: np.ndarray, axis=0) -> None:
        """
        Set normalizer mean and variance to fit the data.
        If data is entered as column vector use axis 1
        IF data is entered as row vector use axis 0
        """
        self.mean = np.mean(X, axis=axis)
        self.mean = self.mean.reshape(self.mean.shape[0], 1)
        self.std = np.std(X - self.mean, axis=axis)
        self.std = self.std.reshape(self.std.shape[0], 1)

    def scale(self, X: np.ndarray):
        """
        Returns normalized input
        """
        return (X - self.mean) / self.std

