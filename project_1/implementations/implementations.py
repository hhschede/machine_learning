import numpy as np
from helpers import *
from pca import *
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def compute_loss(y, tx, w, method = "MSE"):
    """Calculate the loss."""

    if method == "MSE":
        err = y - tx.dot(w)
        loss = 1/2*np.mean(err**2)
        
    elif method == "MAE":  
        err = y - tx.dot(w)
        loss = np.mean(np.abs(err))
        
    elif method == "logistic":
        err = y - sigmoid(tx.dot(w))
        loss = 0.5*np.mean(err**2)
    
    return loss

# -----------------------------------------------------------------------------

def compute_gradient(y, tx, w, lambda_=0, method = "MSE"):
    """Compute the gradient."""
    
    err = y - tx.dot(w)
    if method == "MSE":
        grad = -tx.T.dot(err)/len(err)
        
    elif method == "MAE":
        grad = -tx.T.dot(np.sign(e))/len(err)
        
    elif method == "logistic":
        z = sigmoid(y * tx.dot(w))
        z_ = (z - 1) * y
        grad = (tx.T.dot(z_) + lambda_ * w)/len(y)
        
    return grad

# -----------------------------------------------------------------------------

def compute_hessian(y, tx, w, lam):
    """Returns the hessian of the loss function."""
    
    z = sigmoid(tx.dot(w))
    D = z * (1-z)
    XD = tx * D.reshape(-1,1)
    hess = (tx.T.dot(XD) + lam * np.eye(tx.shape[1]))/len(y)
    
    return hess

# -----------------------------------------------------------------------------

def build_model_data(x, y):
    """Add column for bias term (adding a column of ones to the data set)."""
    
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    
    return y, tx

# -----------------------------------------------------------------------------


def build_poly(x, degree):
    """Polynomial basis functions for input data x."""
    
    poly = x
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
 
    return poly

# -----------------------------------------------------------------------------

def build_interact_terms(x):
    """Calculates interaction terms of second order for each feature."""
    
    interact = (x[:,1:].T * x[:,0]).T
    for i in range(1, x.shape[1]):
        interact = np.c_[interact, (x[:,(i+1):].T*x[:,i]).T]
        
    return interact

# -----------------------------------------------------------------------------

def build_three_way(x):
    """Calculates three-way interaction terms for each feature.
    Note that this function does not do every combination, it only makes a subset
    of all the three-way interaction terms. The reason why is because it takes a 
    lot of time to build these interaction terms and it also increases dramatically
    the number of features, which then can result in memory errors, so we decided
    to compute only a subset of these terms."""

    interact = []
    result = x
    for i in range(x.shape[1]):
        interact = (x[:,(i+1):].T*x[:,i]).T
        for j in range(interact.shape[1]):
            result = np.c_[result, interact[:,j]*x[:,(i+j+1)]]
                
    return result

# -----------------------------------------------------------------------------

def sigmoid(z):
    """ Sigmoid function for logistic regression """
    
    idx = z > 0
    sig = np.zeros(z.shape)
    sig[idx] = 1. / (1 + np.exp(-z[idx]))
    sig[~idx] = np.exp(z[~idx]) / (1. + np.exp(z[~idx]))
        
    return sig

# -----------------------------------------------------------------------------

def predict_labels_logistic(weights, data, threshold = 0.5):
    """Generates class predictions given weights, and a test data matrix"""
    
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred < threshold)] = -1
    y_pred[np.where(y_pred >= threshold)] = 1
    
    return y_pred

# -----------------------------------------------------------------------------

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

# -----------------------------------------------------------------------------

def cross_val(y, x, k):
    """Get subsets for cross validation"""
    
    k_indices = build_k_indices(y, k, 0)

    for i in np.arange(k):
        te_indice = k_indices[i]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
        tr_indice = tr_indice.reshape(-1)
        y_te = y[te_indice]
        y_tr = y[tr_indice]
        x_te = x[te_indice]
        x_tr = x[tr_indice]
        
        # Standardize the sets
        x_train, mean, variance = standardize(x_tr)
        x_test = standardize_test(x_te, mean, variance)
        
        # Apply PCA
        eigVal, eigVec, sumEigVal = PCA(x_train, threshold = 0.90)
        x_train = x_train.dot(eigVec)
        x_test = x_test.dot(eigVec)
        
        # Add a column of ones
        y_tr, x_train = build_model_data(x_train, y_tr)
        y_te, x_test = build_model_data(x_test, y_te)
        
        yield np.array(y_tr), np.array(x_train), np.array(y_te), np.array(x_test) #this is a generator! call next(object) for next set

# -----------------------------------------------------------------------------
        
def do_cross_val(methodtype, k, w, ytrain, xtrain, ytest, xtrainstd, xteststd, lambda_ = 0):
    """do_cross_val function returns a list of cross validation accuracies and a test set accuracy
    
    method type: string, possible inputs 'logistic_regression', 
    k: int of cross validation folds
    w: np array of initial w, must be as long as the training sets
    ytrain: vector of target values from training set
    xtrain: data matrix samples in rows and features in columns
    ytest: vector of target values from test set
    xtrainstd: standardized datamatrix for training w
    xteststd: standardized datamatrix using parameters calculated from xtrainstd
    """
    
    # cross validation for logistic regression
    if methodtype == 'logistic_regression':
    
        gen = cross_val(ytrain, xtrain, k) # initiate generator object
        accuracies = []
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen) # take next subtraining and subtest sets
            w_ = np.random.rand(x_tr.shape[1])
            losses, losses_t, acc, acc_t, w = logistic_hessian(y_tr, x_tr, y_te, x_te, w_, 0.07, 500, 150, writing=False)
            accuracies.append(acc_t[-1])
    
        # Perform algorithm on entire training set to retrieve w
        
        losses, losses_t, acc, test_accuracy, w = logistic_hessian(ytrain, xtrainstd, ytest,
                                                           xteststd, w, 0.07, 500, 150, writing=False)
        # test_accuracies_log.append(test_accuracy[-1])
    
        return accuracies, test_accuracy[-1]

    if methodtype == 'ridge_regression':

        gen = cross_val(ytrain, xtrain, k)
        accuracies = []
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen) # take next subtraining and subtest sets
            w = ridge_regression(y_tr, x_tr, lambda_)

            # Append test accuracies            
            test_pred_lab = predict_labels(w, x_te)
            accuracies.append(pred_accuracy(test_pred_lab, y_te))

        w = ridge_regression(ytrain, xtrainstd, lambda_)
        test_pred_lab = predict_labels(w, xteststd)
        test_accuracy = pred_accuracy(test_pred_lab, ytest)

        return accuracies, test_accuracy
    
# -----------------------------------------------------------------------------
    
def create_dataset(rawdata, rawlabels, interact=False, degree=0, pcathresh=0.9, split=0.8):
    
    """Create splits of data from raw to use for model learning. Adds interact terms and degrees optional
    performs pca to compress dataframe
    
    rawdata = np matrix as it is loaded from train_csv
    rawlabels = values retrieved from train_csv that are binary
    interact = boolean indicating to add interacting terms
    degree = integer indicating how many degrees to add of original training features
    pcathresh = float of percent of variance of data to represent
    split = float of percent of training set to use for model training. 0.8 means 80% of data is used for training
    """
    
    # Create train/test split

    X_train, y_train, X_test, y_test = split_data(rawdata, rawlabels, ratio=split, seed=0)

    # Add interaction terms and polynomial of degree 2 to cross val set (no bias term is added here
    # since its done inside crossval function)
    
    if interact:
        X_train_int = build_interact_terms(X_train)
        X_test_int = build_interact_terms(X_test)
        
        if degree != 0:
            X_train_poly = build_poly(X_train, degree)
            X_test_poly = build_poly(X_test, degree)
            X_train = np.c_[X_train_poly, X_train_int]
            X_test = np.c_[X_test_poly, X_test_int]
        
        else:
            X_train = X_train_int
            X_test = X_test_int
    
    if (interact==False) and (degree != 0):
        X_train = build_poly(X_train, degree)
        X_test = build_poly(X_test, degree)
        
    # Create standardizations for the split
    X_train_std, mean, variance = standardize(X_train)
    X_test_std = standardize_test(X_test, mean, variance)


    # Perform PCA
    eigVal, eigVec, sumEigVal = PCA(X_train_std, threshold = pcathresh)
    X_train_std = X_train_std.dot(eigVec)
    X_test_std = X_test_std.dot(eigVec)

    # Add the bias term for the final models
    y_train, X_train_std = build_model_data(X_train_std, y_train)
    y_test, X_test_std = build_model_data(X_test_std, y_test)

    return y_train, X_train, y_test, X_train_std, X_test_std

# -----------------------------------------------------------------------------

def boxplotter(boxvalues, testvalues, labels, xlabel, ylabel, savefigas):
    
    """boxvalues are the cross validation lists - a list of lists
    testvalues are a list of floats indicating the test accuracies
    labels (list of str) are a list of strings for what the names of each box should be
    xlabel (str)  is the x label of the entire plot
    ylabel (str) is the y label
    savefigas (str) is the name of the image to save to the directory"""
    
    plt.style.use('seaborn')
    plt.boxplot(boxvalues, labels = labels)
    plt.xlabel(xlabel, weight='bold', fontsize=15)
    plt.ylabel(ylabel, weight='bold', fontsize=15)
    
    for i in range(len(testvalues)):
        plt.plot(i+1, testvalues[i], marker='o', c='red')

    plt.savefig(savefigas)

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# LEARNING ALGORITHM FUNCTIONS
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# LINEAR REGRESSION
# -----------------------------------------------------------------------------------

def least_squares_GD(y, tx, y_t, tx_t, initial_w, tol = 1e-5, max_iters = 10000, gamma = 0.05, k=4,  write = False):
    """Gradient descent algorithm with option for cross validation. 
    
    For cross validation: If k is set to > 0, then the function returns
    the average test loss between the trials, the variance between the trials, the vector of test losses over 
    gradient descent (for each cross validation, so it is a list of lists), the training loss and the 
    last w that was retrieved from the last trial (if one wants to use it for plotting).
    
    For retrieving the optimal w: If k is set to 0, the entire y and tx values given to the function will
    be used to calculate the optimal w. the only value that will be returned is w."""
    
    test_loss = []
    test_losses_vector = []
    training_loss = []

    # This if is for a final model, that is - if we want to get the best w's (so use all training samples)
    
    if k == 0:
        
        w = initial_w
        ws = [initial_w]
        
        losses_tr = [compute_loss(y, tx, w)]
        losses_ts = [compute_loss(y_t, tx_t, w)]
        
        acc_tr = [pred_accuracy(predict_labels(w, tx), y)]
        acc_ts = [pred_accuracy(predict_labels(w, tx_t), y_t)]
        
        for i in range(max_iters):
            
            # Compute gradient
            gd = compute_gradient(y, tx, w)
            
            # Compute next w
            w = w - gamma*gd
            
            ws.append(w)
            losses_tr.append(compute_loss(y, tx, w))
            losses_ts.append(compute_loss(y_t, tx_t, w))
            
            acc_tr.append(pred_accuracy(predict_labels(w, tx), y))
            acc_ts.append(pred_accuracy(predict_labels(w, tx_t), y_t))
            
            if write and (i % 100 == 0):
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=i, ti=max_iters - 1, l=losses_tr[-1], w0=w[0], w1=w[1]))
            
            
        return losses_tr, losses_ts, acc_tr, acc_ts, ws
    
    # This else is for determining if this is a good way to select the model - return test and training errors
    else:
        
        gen = cross_val(y, tx, k) # Initiate generator object
        w_final = []
        accuracies = []
        
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen)

            # Define parameters to store w and loss
            
            w = initial_w
            
            my_losses = [compute_loss(y_tr, x_tr, w)]
            test_mylosses = []
            loss = my_losses[0]
            
            n_iter=0
            
            while n_iter < max_iters:
                
                # Compute gradient
                gd = compute_gradient(y_tr, x_tr, w)
                
                # Compute next w
                w = w - gamma*gd
                
                # Compute loss and diff
                loss = compute_loss(y_tr, x_tr, w)
                
                # Store w and loss and increment
                n_iter += 1
                my_losses.append(loss)
                test_mylosses.append(compute_loss(y_te, x_te, w))
                
                if abs(my_losses[-1] - my_losses[-2]) < tol:
                    continue
                    
                if write == True:
                    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
                    
            test_losses_vector.append(test_mylosses)
            
            # Append the list of test accuracies
            w_final.append(w)
            test_pred_lab = predict_labels(w, x_te)
            accuracies.append(pred_accuracy(test_pred_lab, y_te))

            # Append the test_loss to the list so it can be averaged at the end
            test_loss.append(compute_loss(y_te, x_te, w))
            training_loss.append(loss)
            
        return np.array(test_loss).mean(), np.array(test_loss).var(), np.array(test_losses_vector), np.array(training_loss).mean(), w_final, accuracies

# -----------------------------------------------------------------------------

def least_squares_SGD(y, tx, y_t, tx_t, initial_w, batch_size = 1000, tol = 1e-4,patience = 1, max_iters = 1000, gamma = 0.05, write = False):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w)]
    losses_ts = [compute_loss(y_t, tx_t, w)]
    
    acc_tr = [pred_accuracy(predict_labels(w, tx), y)]
    acc_ts = [pred_accuracy(predict_labels(w, tx_t), y_t)]
    
    diff = losses_tr[0]
    n_iter=0
    nb_ES = 0
    
    while (n_iter <= max_iters) and (nb_ES < patience): 
        for mini_y, mini_tx in batch_iter(y, tx, batch_size):
            
            # Compute gradient
            gd = compute_gradient(mini_y, mini_tx, w)
            
            # Compute next w
            w = w - gamma*gd
            
            # Compute loss
            losses_tr.append(compute_loss(y, tx, w))
            losses_ts.append(compute_loss(y_t, tx_t, w))
            
            # Compute accuracy
            acc_tr.append(pred_accuracy(predict_labels(w, tx), y))
            acc_ts.append(pred_accuracy(predict_labels(w, tx_t), y_t))
                      
            diff = abs(losses_tr[-2]-losses_tr[-1])
            
            n_iter += 1
            if (diff < tol):
                nb_ES = nb_ES + 1
            else:
                nb_Es = 0
            
        if write and (n_iter % 500 == 0):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))
            
    return losses_tr, losses_ts, acc_tr, acc_ts, w


# -----------------------------------------------------------------------------------
# LEAST SQUARES
# -----------------------------------------------------------------------------------


def least_squares(y, tx, k=0):
    """Calculate the least square solution with cross validation option
    
    Setting k == 0 will return the optimal w vector given the entire dataset fed to the function
    
    Setting k > 0 will split the dataset given to the function into k folds to perform CV on. This
    will result in returning the following values:
        (the mean of the test losses, the variance of the test losses, 
        the mean of the training set losses and the last value for w that was calculated)."""
    
    test_loss = []
    training_loss = []
    
    
    # If k == 0, the function returns the best w given the entire dataset used as input
    if k ==0:
        
        X = tx.T.dot(tx)
        Y = tx.T.dot(y)
        w = np.linalg.lstsq(X,Y)[0]
        return w

    # If k > 0, determine if this is a good way to select the model (retrieve variance and bias)
    else:
        
        w_final = []
        accuracies = []
        gen = cross_val(y, tx, k) # Initiate generator object
        
        for i in np.arange(k):
            
            y_tr, x_tr, y_te, x_te = next(gen) # Take next subtraining and subtest sets
            X = x_tr.T.dot(x_tr)
            Y = x_tr.T.dot(y_tr)
            w = np.linalg.lstsq(X,Y)[0]
            
            test_loss.append(compute_loss(y_te, x_te, w))
            training_loss.append(compute_loss(y_tr, x_tr, w))
            
            # Append test accuracies
            w_final.append(w)
            test_pred_lab = predict_labels(w, x_te)
            accuracies.append(pred_accuracy(test_pred_lab, y_te))
        return np.array(test_loss).mean(), np.array(test_loss).var(), np.array(training_loss).mean(), w_final, accuracies
    

# -----------------------------------------------------------------------------------
# RIDGE REGRESSION
# -----------------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Calculates ridge regression."""
    
    xTx = tx.T.dot(tx)
    X = xTx + (2 * len(y) * lambda_ * np.identity(len(xTx)))
    Y = tx.T.dot(y)
    
    return np.linalg.lstsq(X, Y)[0]


# -----------------------------------------------------------------------------------
# LOGISTIC REGRESSION
# -----------------------------------------------------------------------------------


def logistic_regression(y, tx, y_t, tx_t, initial_w, max_iters = 10000, gamma = 0.0005, method = "gd",batch_size = 1000, writing = False):
    """ Simple logistic regression model function 
        Takes parameters for optimazation:
        max_iters = number of iterations
        gamma = learning rate
        method = gradient descent (gd) or stochastic gradient descent (sgd)
        batch_size = size of batch for sgd
        writing = boolean if true write prograssion of learning """
    
    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w, lam = 0, method = 'logistic')]
    losses_ts = [compute_loss(y_t,tx_t,w,lam= 0,method = 'logistic')]
    
    pred = predict_labels_logistic(w, tx)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    if method == "gd":
        
        for i in range(max_iters):
            
            # Compute gradient
            grad = compute_gradient(y, tx, initial_w, lambda_=0, method = "logistic")
            
            # Compute next w
            w -= gamma * grad
            
            # Append loss
            losses_tr.append(compute_loss(y, tx, w, lam = 0, method = 'logistic'))
            losses_ts.append(compute_loss(y_t,tx_t,w,lam=0,method = 'logistic'))
            
            # Append accuracies
            pred = predict_labels_logistic(w, tx)
            acc_tr.append(pred_accuracy(pred, y))
            
            pred = predict_labels_logistic(w, tx_t)
            acc_ts.append(pred_accuracy(pred, y_t))
            
            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss_ts : {l}".format(
                        iter = i, l=losses_ts[-1] ))
    if method == "sgd":
        
        for i in range(max_iters):   
            for mini_y, mini_X in batch_iter(y, tx, batch_size): 
                
                # Compute gradient
                grad = compute_gradient(y, tx, initial_w,lambda_=0, method = "logistic")

                # Compute next w
                w -= gamma * grad
                
                # Compute losses
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = 0, method = "logistic"))
                losses_ts.append(compute_loss(y_t,tx_t,w,lam=0,method = 'logistic'))
                
                # Compute accuracies
                pred = predict_labels_logistic(w, tx)
                acc_tr.append(pred_accuracy(pred, y))
                
                pred = predict_labels_logistic(w, tx_t)
                acc_ts.append(pred_accuracy(pred, y_t))

            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss_ts : {l}".format(
                        iter = i, l=losses_ts[-1] ))
            
    return losses_tr, losses_ts, acc_tr, acc_ts, w

# -----------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, y_t, tx_t, initial_w, lamb = 0.01, max_iters = 10000, gamma = 0.0005, method = "gd", batch_size = 1000, writing = False):
       
    """ Regularized logistic regression model function 
        Takes parameters for optimazation:
        lamda = regularization parameter
        max_iters = number of iterations
        gamma = learning rate
        method = gradient descent (gd) or stochastic gradient descent (sgd)
        batch_size = size of batch for sgd
        writing = boolean if true write prograssion of learning."""
    
    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w, lam = lamb, method = 'logistic')]
    losses_ts = [compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic')]
    
    pred = predict_labels_logistic(w, tx)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    if method == "gd":
        
        for i in range(max_iters):
            
            # Compute gradient
            grad = compute_gradient(y, tx, initial_w, lambda_=lamb, method = "logistic")
            
            # Compute next w
            w -= gamma * grad
            
            # Compute losses
            losses_tr.append(compute_loss(y, tx, w, lam = lamb, method = 'logistic'))
            losses_ts.append(compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic'))
            
            # Compute accuracies
            pred = predict_labels_logistic(w, tx)
            acc_tr.append(pred_accuracy(pred, y))
            
            pred = predict_labels_logistic(w, tx_t)
            acc_ts.append(pred_accuracy(pred, y_t))
            
            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss_ts : {l}".format(
                        iter = i, l=losses_ts[-1] ))
                    
    if method == "sgd":
        
        for i in range(max_iters):   
            for mini_y, mini_X in batch_iter(y, tx, batch_size):   
                
                # Compute gradient
                grad = compute_gradient(y, tx, initial_w,lambda_=lamb, method = "logistic")

                # Compute next w
                w -= gamma * grad
                ws[i+1] = w
                
                # Compute losses
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = lamb, method = "logistic"))
                losses_ts.append(compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic'))
                
                # Compute accuracies
                pred = predict_labels_logistic(w, tx)
                acc_tr.append(pred_accuracy(pred, y))
                
                pred = predict_labels_logistic(w, tx_t)
                acc_ts.append(pred_accuracy(pred, y_t))

            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss_ts : {l}".format(
                        iter = i, l=losses_ts[-1] ))
            
    return losses_tr, losses_ts, acc_tr, acc_ts, w

# -----------------------------------------------------------------------------------

def logistic_hessian(y, tx, y_t, tx_t, initial_w, gamma=0.05, lam=0.1, max_iters = 100, momentum = 0.9, tol=1e-8, patience = 1, writing = True, threshold = 0.5):
    """ Regularized/simple logistic regression computed with Netwon's method
    
        Takes parameters for optimazation:
        lam = lambda (regularization parameter)
        max_iters = number of iterations
        gamma = learning rate
        method = gradient descent (gd) or stochastic gradient descent (sgd)
        batch_size = size of batch for sgd
        writing = boolean if true write prograssion of learning
        
        tol and patience for early stopping. 
        The learning stop after the difference in loss is small than the tolerence (tol) for a number of times (patience)
        
        momentum = value for modifying the learning rate using momentum method. """
    
    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w, method = "logistic")]
    losses_ts = [compute_loss(y_t, tx_t, w, method = "logistic")]
    
    pred = predict_labels_logistic(w, tx, threshold)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t, threshold)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    diff = 10
    n_iter = 0
    nb_ES = 0
    velocity = 0
    
    while (n_iter < max_iters) and (patience > nb_ES):
        
        # Compute gradient
        gd = compute_gradient(y, tx, w, lam, method = "logistic")
        
        # Compute hessian
        hess = compute_hessian(y, tx, w, lam)
        
        # Compute next w
        velocity = momentum*velocity - gamma*np.linalg.solve(hess,gd)
        w = w + velocity
        
        # Compute loss and accuracies
        losses_tr.append(compute_loss(y, tx, w, method = "logistic"))
        pred = predict_labels_logistic(w, tx, threshold)
        acc_tr.append(pred_accuracy(pred, y))
        
        losses_ts.append(compute_loss(y_t, tx_t, w, method = "logistic"))
        pred_t = predict_labels_logistic(w, tx_t, threshold)
        acc_ts.append(pred_accuracy(pred_t, y_t))
        
        # Compute diff
        diff = abs(losses_tr[-2]-losses_tr[-1])
        
        n_iter += 1
        
        if writing:
            if n_iter % 25 == 0:
                print("{0}/{1}\t train acc : {2} \t | test acc : {3}".format(n_iter, max_iters, acc_tr[-1],acc_ts[-1]))
        
        if (diff < tol):
                nb_ES = nb_ES + 1
        else:
            nb_ES = 0
            
    return losses_tr, losses_ts, acc_tr, acc_ts, w

# -----------------------------------------------------------------------------------

def Grid_Search_logistic(y, tx, y_t, tx_t, initial_w, gamma=0.05, lam=0.1, max_iters = 100, momentum = 0, tol=1e-5, patience = 5):
    """Simpified version of logistic regressiong for faster computing of grid search."""

    w = initial_w
    
    pred_t = predict_labels_logistic(w, tx_t, 0.5)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    velocity = 0
    
    diff = 10
    n_iter = 0
    nb_ES = 0
    
    while (n_iter < max_iters) and (patience > nb_ES):
        
        # Compute gradient
        gd = compute_gradient(y, tx, w, lam, method = "logistic")
        
        # Compute hesssian
        hess = compute_hessian(y, tx, w, lam)
        
        # Compute next w
        velocity = momentum*velocity - gamma*np.linalg.solve(hess,gd)
        w = w + velocity
        
        # Compute accuracy
        pred_t = predict_labels_logistic(w, tx_t, 0.5)
        acc_ts.append(pred_accuracy(pred_t, y_t))

        n_iter = n_iter + 1
        if (abs(acc_ts[-1]-acc_ts[-2]) < tol):
                nb_ES = nb_ES + 1
        else:
            nb_ES = 0
    
    return max(acc_ts)
