import numpy as np

class Dataloader():
    """
    Dataloader class utilized to create lists of labeled numpy matrices
    
    Each iteration on the dataloader object returns a matrix containing 
    features and labeled outputs for those features

    Warning: Make sure you input both X and Y as column vectors

    """
    X: np.ndarray    
    Y: np.ndarray
    n_batches: int
    batch_size: int
    index: int = 0

    def __init__(self, X, Y, n_batches):
        """
        inputs
        ---
        X: Matrix of features X as column vectors
        Y: Output labels Y as column vectors
        n_batches: number of batches you want to have for the entire data
        """
        self.X = X
        self.Y = Y
        self.n_batches = n_batches
        self.validate_data()
        self.batch_size = int(self.X.shape[1] / self.n_batches)

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.X.shape[1]

    def __next__(self):
        position = self.index * self.batch_size
        if position >= len(self):
            raise StopIteration
        x = self.X[:, position:position+self.batch_size]
        y = self.Y[:, position:position+self.batch_size]
        self.index += 1
        return x, y

    def __repr__(self):
        return f"Dataloader(X: {self.X.shape}, Y: {self.Y.shape}, n_batches: {self.n_batches}, batch_size: {self.batch_size})"

    def validate_data(self):
        """
        Runs input data validations
        """
        assert type(self.X) == np.ndarray, "X is not a numpy array"
        assert type(self.Y) == np.ndarray, "Y is not a numpy array"
        assert self.X.shape[1] == self.Y.shape[1], "X and Y do not have the same number of columns/items"
        assert self.X.shape[1] % self.n_batches == 0, f"X({self.X.shape[1]}) is not divisible by {self.n_batches}"
        assert self.Y.shape[1] % self.n_batches == 0, f"X({self.Y.shape[1]}) is not divisible by {self.n_batches}"
    