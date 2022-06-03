import numpy as np
import pandas as pd
import dill
from redmind.network import NeuralNetwork

def one_hot_encode(position: int, num_classes: int):
    """
    One hot encodes a given class into a (1, num_classes) row vector
    """
    zero_vector = np.zeros((1, num_classes), dtype=int) 
    zero_vector[0][position] = 1
    # return 1d row vector
    return zero_vector.reshape(-1)

def save_model(nn: NeuralNetwork, filename: str = 'nn.dill') -> None:
    if isinstance(nn, NeuralNetwork):
        print(f"Saving Neural Network into {filename}")
        with open(filename, 'wb') as f:
            dill.dump(nn, f)
    else:
        print("Please provide a valid Neural Network")
    return None

def load_model(filename: str = 'nn.dill') -> NeuralNetwork:
    print(f"Loading NN {filename}")
    with open(filename, 'rb') as f:
        nn = dill.load(f)
    return nn

def split_dataframe(dataframe: pd.core.frame.DataFrame, y_col_idx = -1, train_percent=80, y_one_hot_encode = False, num_classes = 0, shuffle=False) -> np.ndarray:
    """
    Splits pandas dataframe into 3 groups train, dev and test sets and returns datasets as numpy arrays
    It returns the following
    X_train, Y_train / X_dev, Y_dev / X_test, Y_test
    Usage:  X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_dataframe(dataframe, y_col_idx = 5, train_percent=80)

    Arguments
    dataframe: Dataframe containing X and Y labels. dataframe is expected as row vectors
    y_col_idx: Y label column index. If no argument is passed then it will asume y label is the last column (-1)
    train_percent: percentage of data used for training set. dev and test sets will get the remaining split by 2.
                   for example if train_percent=60, then dev and test will get 20 and 20 percent respectively
    shuffle: Wether or not to shuffle the data randomly
    y_one_hot_encode: If True all Y outputs will be one hot encoded
    num_classes: If y_one_hot_encode is enabled you will need to set this to the number of classes
    """
    assert type(dataframe) == pd.core.frame.DataFrame, "Data should be entered as pandas Dataframe"
    assert train_percent > 50 and train_percent < 100, "train_percent should be between 50 and 100"
    print(f"Entered {dataframe.shape[0]} rows")
    total_samples = dataframe.shape[0]
    train_samples = int(total_samples * train_percent / 100)
    dev_samples = int((total_samples - train_samples) / 2)
    test_samples = int(total_samples - train_samples - dev_samples)
    if shuffle:
        dataframe = dataframe.sample(frac=1)
    print(f"Data split: train: {train_samples}, dev: {dev_samples}, test: {test_samples}")
    # remove Y label column first
    dataset_y = dataframe.iloc[:, y_col_idx]
    dataset_x = dataframe.drop(dataframe.columns[y_col_idx], axis=1)

    X_train = dataset_x.iloc[:train_samples].to_numpy()
    Y_train = dataset_y.iloc[:train_samples].to_numpy()

    X_dev = dataset_x.iloc[train_samples:train_samples + dev_samples].to_numpy()
    Y_dev = dataset_y.iloc[train_samples:train_samples + dev_samples].to_numpy()
    
    X_test = dataset_x.iloc[total_samples - test_samples:].to_numpy()
    Y_test = dataset_y.iloc[total_samples - test_samples:].to_numpy()
    # One hot encode Y's
    if y_one_hot_encode:
        assert num_classes > 0, "num_classes should be above 0"
        Y_train = np.array([one_hot_encode(i, num_classes) for i in Y_train])
        Y_dev = np.array([one_hot_encode(i, num_classes) for i in Y_dev])
        Y_test = np.array([one_hot_encode(i, num_classes) for i in Y_test])
    # else return Y's as column vectors
    else:
        Y_train = Y_train.reshape(Y_train.shape[0], 1)
        Y_dev = Y_dev.reshape(Y_dev.shape[0], 1)
        Y_test = Y_test.reshape(Y_test.shape[0], 1)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test