import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


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


def test_train(X, y, test_ratio=0.3):
    """The test_train function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    the input X is a numpy matrix with samples in rows and features in columns.
    the input y is a numpy array with entry corresponding to the target value
    of the corresponding sample in X. This may be a class or a float.
    The test_ratio is a float, default 0.3, that sets the test set fraction of
    the entire dataset to 0.3 and keeps the other part for training"""

    import random
    import numpy as np

    # Perform shuffling
    idx_shuffled = np.random.permutation(np.arange(0, X.shape[0]))
    # return shuffled X and y
    X_shuff = []
    y_shuff = []
    for sample in idx_shuffled:
        X_shuff.append(X[sample])
        y_shuff.append(y[sample])
    X_shuff = np.matrix(X_shuff)
    y_shuff = np.array(y_shuff)

    # Cut the data set into train and test
    test_num = int(X.shape[0] * test_ratio)
    train_num = int(X.shape[0] - test_num)

    # print(X_shuff)
    X_train = X_shuff[0:train_num]
    y_train = y_shuff[0:train_num]
    X_test = X_shuff[train_num:]
    y_test = y_shuff[train_num:]

    return X_train, y_train, X_test, y_test


def pred_accuracy(predict, y):
    trues = []
    for num, val in enumerate(y):
        if int(val) == predict[num]:
            trues.append(1)
        else:
            trues.append(0)
    trues = np.array(trues)
    percent = trues.mean()
    #print(percent)
    return percent

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

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