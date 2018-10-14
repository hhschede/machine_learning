import csv
import numpy as np

# -----------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------

def build_model_data(data, labels):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(labels)
    tx = np.c_[np.ones(num_samples), data]
    return labels, tx

# -----------------------------------------------------------------------------------

def predict_labels(weights, data, threshold = 0.5):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    
    return y_pred

# -----------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------

def split_data(X, y, ratio=0.8, seed=1):
    """The split_data function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    The input X is a numpy array with samples in rows and features in columns.
    The input y is a numpy array with each entry corresponding to the label
    of the corresponding sample in X. This may be a class or a float.
    The ratio variable is a float, default 0.8, that sets the train set fraction of
    the entire dataset to 0.8 and keeps the other part for test set"""
    
    import random
    
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

# -----------------------------------------------------------------------------------

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# -----------------------------------------------------------------------------------

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """get subsets for cross validation"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    return y_tr, x_tr, y_te, x_te

# -----------------------------------------------------------------------------------

def pred_accuracy(predict, y):
    compare = (predict == y)
    percent = compare.mean()
    return percent

# -----------------------------------------------------------------------------------

def standardize(x):
    """Standardize the original data set."""
    mean = np.mean(x, axis=0)
    center = x - mean
    variance = np.std(center, axis=0)
    standard_data = center / variance
    return standard_data, mean, variance

# -----------------------------------------------------------------------------------

def standardize_test(x, means, variance)
    """Standardize the test set using training set means and variance"""
    center = x - means
    standard_data = center / variance
    return standard_data

# -----------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------
            
def process_data(data, data_t, labels, ids, zero = False, sample_filtering = True, feature_filtering = True):
    """The process_data function prepares the train and test data by performing
    some data cleansing techniques. Missing values which are set as -999
    are replaced by NaN, then the means of each features are calculated
    in order to replace NaN values by the means. Features and samples with
    too much missing values are deleted by this function. Returns filtered
    dataset and column idx to be removed as well as column means and variance
    (required for transformation of unseen data."""

    data_process = np.array(data[:,:])
    data_p_t = np.array(data_t[:,:])
    lab = np.array(labels[:])
    
    if zero == True:
        # Setting labels as 0 (background) or 1 (signal)
        lab[labels == -1] = 0
    
    # Setting missing values (-999) as NaN
    data_process[data_process == -999] = np.nan
    data_p_t[data_p_t == -999] = np.nan

    # Filtering weak features and samples

    # Retrieving percentage for each feature 
    
    if feature_filtering:
        nan_count = np.count_nonzero(np.isnan(data_process),axis=0)/data_process.shape[0]
        
        # Filter out features which have NaN values higher than 60%
        data_set_filtered = []
        data_t_filtered = []
        for idx, entry in enumerate(nan_count):
            if entry < 0.6:
                # Append the column of the original dataset that is good
                data_set_filtered.append(data_process.T[idx])
                data_t_filtered.append(data_p_t.T[idx])
        data_set_filtered = np.array(data_set_filtered).T
        data_t_filtered = np.array(data_t_filtered).T
        
    if sample_filtering:
        data_set_filtered_2 = [] # Dataset filtered for features and samples
        
        # Arrays that gets rid of entries that are no longer corresponding in the dataframe
        y = [] 
        ids_filt = [] 
        
        # Retrieve percentage for each sample
        nan_count_2 = np.count_nonzero(np.isnan(data_set_filtered),axis=1)/data_set_filtered.shape[1]
        
        # Gets rid of samples with NaN values higher than 15%
        for idx, entry in enumerate(nan_count_2):
            if entry < 0.15:
                y.append(lab[idx])
                ids_filt.append(ids[idx])
                data_set_filtered_2.append(data_set_filtered[idx])
        data_set_filtered_2 = np.array(data_set_filtered_2)
        y = np.array(y)


    # Print new dimensions of the dataframe after filtering

    print('The original dimensions of the training data set was {0} samples'
          ' and {1} columns. After feature and sample filtering, there are'
          ' {2} samples and {3} columns'.format(data_process.shape[0],
                                                data_process.shape[1],
                                                data_set_filtered_2.shape[0],
                                                data_set_filtered_2.shape[1]))

    # Getting Rid of NaN and Replacing with mean

    data_nan = data_set_filtered_2.copy()
    data_t_nan = data_t_filtered.copy() 
    # Create list with average values of columns, excluding NaN values
    column_means = np.nanmean(data_nan, axis=0)
    column_means_t = np.nanmean(data_t_nan, axis=0)
    
    # Variable containing locations of NaN in data frame
    inds = np.where(np.isnan(data_nan)) 
    inds_t = np.where(np.isnan(data_t_nan)) 
    
    # Reassign locations of NaN to the column means
    data_nan[inds] = np.take(column_means, inds[1])
    data_t_nan[inds_t] = np.take(column_means_t, inds_t[1])
    return (data_nan, data_t_nan, y, ids_filt)
