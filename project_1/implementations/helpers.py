import csv
import numpy as np
import random
import matplotlib.pyplot as plt

from implementations import *

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # Sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

# -----------------------------------------------------------------------------

def predict_labels(weights, data, threshold = 0):
    """Generates class predictions given weights, and a test data matrix"""
    
    y_pred = (np.dot(data, weights))
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    
    return y_pred

# -----------------------------------------------------------------------------

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

# -----------------------------------------------------------------------------

def split_data(X, y, ratio=0.8, seed=1):
    """The split_data function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    The input X is a numpy array with samples in rows and features in columns.
    The input y is a numpy array with each entry corresponding to the label
    of the corresponding sample in X. This may be a class or a float.
    The ratio variable is a float, default 0.8, that sets the train set fraction of
    the entire dataset to 0.8 and keeps the other part for test set"""
        
    # Set seed
    np.random.seed(seed)

    # Perform shuffling
    idx_shuffled = np.random.permutation(len(y))
    
    # Return shuffled X and y
    X_shuff = X[idx_shuffled]
    y_shuff = y[idx_shuffled]

    # Cut the data set into train and test
    train_num = round(len(y) * ratio)
    X_train = X_shuff[:train_num,:]
    y_train = y_shuff[:train_num]
    X_test = X_shuff[train_num:,:]
    y_test = y_shuff[train_num:]

    return X_train, y_train, X_test, y_test

# -----------------------------------------------------------------------------

def pred_accuracy(predict, y):
    """Computes the accuracy of a prediction on the labels."""
    
    compare = (predict == y)
    percent = np.mean(compare)
    
    return percent

# -----------------------------------------------------------------------------

def standardize(x):
    """Standardize the original data set."""
    
    mean = np.mean(x, axis=0)
    center = x - mean
    
    variance = np.std(center, axis=0)
    variance[variance==0] = 1
    
    standard_data = center / variance

    return standard_data, mean, variance

# -----------------------------------------------------------------------------

def standardize_test(x, means, variance):
    """Standardize the test set using training set means and variance"""
    
    center = x - means
    variance[variance==0] = 1

    standard_data = center / variance

    return standard_data

# -----------------------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # so here np.arrange arranges all the number from 0 to data_size-1
        # than the np.randome.permutation will mix all those number
        # example np.random.permutation(np.arange(5)) = 4, 0, 2, 3, 1
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index] 