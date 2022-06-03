import pandas as pd
import numpy as np

def split_dataframe(dataframe: pd.core.frame.DataFrame, y_col_idx = -1, train_percent=80, shuffle=False) -> np.ndarray:
    """
    Splits dataframe into 3 groups train, dev and test sets and returns datasets as numpy arrays
    It returns the following
    X_train, Y_train / X_dev, Y_dev / X_test, Y_test
    Usage:  X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_dataframe(dataframe, y_col_idx = 5, train_percent=80)

    Arguments
    dataframe: Dataframe containing X and Y labels. dataframe is expected as row vectors
    y_col_idx: Y label column index. If no argument is passed then it will asume y label is the last row (-1)
    train_percent: percentage of data used for training set. dev and test sets will get the remaining split by 2.
                   for example if train_percent=60, then dev and test will get 20 and 20 percent respectively
    shuffle: Wether or not to shuffle the data randomly
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
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    X_dev = dataset_x.iloc[train_samples:train_samples + dev_samples].to_numpy()
    Y_dev = dataset_y.iloc[train_samples:train_samples + dev_samples].to_numpy()
    Y_dev = Y_dev.reshape(Y_dev.shape[0], 1)
    X_test = dataset_x.iloc[total_samples - test_samples:].to_numpy()
    Y_test = dataset_y.iloc[total_samples - test_samples:].to_numpy()
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test