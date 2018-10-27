import numpy as np
from helpers import *
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

def compute_loss(y, tx, w, lam = 0, method = "MSE"):
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
        z0 = (z - 1) * y
        grad = (tx.T.dot(z0) + lambda_ * w)/len(y)
    return grad

# -----------------------------------------------------------------------------

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    
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


def build_poly_rt(x, degree):
    """Polynomial basis functions for input data x."""
    
    poly = np.power(abs(x), 1./2)
    for deg in range(3, degree+1):
        poly = np.c_[poly, np.power(x, 1./deg)]
    return poly

# -----------------------------------------------------------------------------

def build_interact_terms(x):
    """Calculates interaction terms for each feature."""
    
    interact = (x[:,1:].T * x[:,0]).T
    for i in range(1, x.shape[1]):
        interact = np.c_[interact, (x[:,(i+1):].T*x[:,i]).T]
    return interact

# -----------------------------------------------------------------------------

def build_three_way(x):
    """Calculates triple interaction terms for each feature."""

    interact = []
    result = x
    for i in range(x.shape[1]):
        interact = (x[:,(i+1):].T*x[:,i]).T
        for j in range(interact.shape[1]):
            result = np.c_[result, interact[:,j]*x[:,(i+j+1)]]
                
    return result

# -----------------------------------------------------------------------------

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# -----------------------------------------------------------------------------------

def cross_val(y, x, k, lambda_, degree):
    """get subsets for cross validation"""
    
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
        # This is a generator! Call next(object) for next set
        yield np.array(y_tr), np.array(x_train), np.array(y_te), np.array(x_test) 

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
    last w that was retrieved from the last trial (if one wants to use it for plotting etc)
    
    For retrieving the optimal w: If k is set to 0, the entire y and tx values given to the function will
    be used to calculate the optimal w. the only value that will be returned is w"""
    
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
            # compute gradient
            gd = compute_gradient(y, tx, w)
            # compute next w
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
        gen = cross_val(y, tx, k, 0.01, 2) # initiate generator object
        w_final = []
        accuracies = []
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen)

            # Define parameters to store w and loss
            # ws = [initial_w]
            w = initial_w
            my_losses = [compute_loss(y_tr, x_tr, w)]
            test_mylosses = []
            loss = my_losses[0]
            n_iter=0
            while n_iter < max_iters:
                # compute gradient
                gd = compute_gradient(y_tr, x_tr, w)
                # compute next w
                w = w - gamma*gd
                # compute loss and diff
                loss = compute_loss(y_tr, x_tr, w)
                # store w and loss and increment
                # ws.append(w)
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

            # append the test_loss to the list so it can be averaged at the end
            test_loss.append(compute_loss(y_te, x_te, w))
            training_loss.append(loss)
            
            
        return np.array(test_loss).mean(), np.array(test_loss).var(), np.array(test_losses_vector), np.array(training_loss).mean(), w_final, accuracies

# -----------------------------------------------------------------------------------

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
            # compute gradient
            gd = compute_gradient(mini_y, mini_tx, w)
            
            # compute next w
            w = w - gamma*gd
            
            # compute loss
            losses_tr.append(compute_loss(y, tx, w))
            losses_ts.append(compute_loss(y_t, tx_t, w))
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
        gen = cross_val(y, tx, k, 0.01, 2) # initiate generator object
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen) # take next subtraining and subtest sets
            X = x_tr.T.dot(x_tr)
            Y = x_tr.T.dot(y_tr)
            try:
                w = np.linalg.lstsq(X,Y)[0]
            except np.linalg.LinAlgError as err: # Due to a singular matrix, need to add some noise for it to work
                X += 0.000001
                Y += 0.000001
                w = np.linalg.solve(X,Y)
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
    return np.linalg.solve(X, Y)



# -----------------------------------------------------------------------------------
# LOGISTIC REGRESSION
# -----------------------------------------------------------------------------------

# USEFUL FUNCTIONS

def sigmoid(z):
    """ Sigmoid function for logistic regression """
    idx = z > 0
    sig = np.zeros(z.shape)
    sig[idx] = 1. / (1 + np.exp(-z[idx]))
    sig[~idx] = np.exp(z[~idx]) / (1. + np.exp(z[~idx]))
        
    return sig

# -----------------------------------------------------------------------------------

def predict_labels_logistic(weights, data, threshold = 0.5):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred < threshold)] = -1
    y_pred[np.where(y_pred >= threshold)] = 1
    
    return y_pred


# -----------------------------------------------------------------------------------

def logistic_regression(y, tx, y_t, tx_t, initial_w, max_iters = 10000, gamma = 0.0005, method = "gd",batch_size = 1000, writing = False):
    """ Simple logistic regression model function 
    Takes parameters for optimazation:
        max_iters = number of iterations
        gamma = learning rate
        method = gradient descent (gd) or stochastic gradient descent (sgd)
        batch_size = size of batch for sgd
        writing = boolean if true write prograssion of learning
    
    """
    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w, lam = 0, method = 'logistic')]
    losses_ts = [compute_loss(y_t,tx_t,w,lam= 0,method = 'logistic')]
    
    pred = predict_labels_logistic(w, tx)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t)
    acc_ts = [pred_accuracy(pred_t, y_t)]

    
    if method == "gd":
        for i in range(max_iters):
            grad = compute_gradient(y, tx, initial_w, lambda_=0, method = "logistic")
            
            w -= gamma * grad

            losses_tr.append(compute_loss(y, tx, w, lam = 0, method = 'logistic'))
            losses_ts.append(compute_loss(y_t,tx_t,w,lam=0,method = 'logistic'))
            
            
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
                grad = compute_gradient(y, tx, initial_w,lambda_=0, method = "logistic")


                w -= gamma * grad
                
                
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = 0, method = "logistic"))
                losses_ts.append(compute_loss(y_t,tx_t,w,lam=0,method = 'logistic'))
                
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
        writing = boolean if true write prograssion of learning
    
    """
    
    w = initial_w
    
    losses_tr = [compute_loss(y, tx, w, lam = lamb, method = 'logistic')]
    losses_ts = [compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic')]
    
    pred = predict_labels_logistic(w, tx)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t)
    acc_ts = [pred_accuracy(pred_t, y_t)]

    
    if method == "gd":
        for i in range(max_iters):
            grad = compute_gradient(y, tx, initial_w, lambda_=lamb, method = "logistic")
            
            w -= gamma * grad

            losses_tr.append(compute_loss(y, tx, w, lam = lamb, method = 'logistic'))
            losses_ts.append(compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic'))
            
            
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
                grad = compute_gradient(y, tx, initial_w,lambda_=lamb, method = "logistic")


                w -= gamma * grad
                
                ws[i+1] = w
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = lamb, method = "logistic"))
                losses_ts.append(compute_loss(y_t,tx_t,w,lam=lamb,method = 'logistic'))
                
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

def compute_hessian(y, tx, w, lam):
    """return the hessian of the loss function."""
    
    z = sigmoid(tx.dot(w))
    D = z * (1-z)
    XD = tx * D.reshape(-1,1)
    hess = tx.T.dot(XD) + lam*np.eye((tx.shape[1]))
    return hess/len(y)


def logistic_hessian(y, tx, y_t, tx_t, initial_w, gamma=0.05, lam=0.1, max_iters = 100, momentum = 0.5, tol=1e-8, patience = 1, writing = True, threshold = 0.5):
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
        
        momentum = value for modifying the learning rate using momentum method
    
    """
    # Define parameters to store w
    w = initial_w
    # ws = [w]
    
    # Compute losses
    losses_tr = [compute_loss(y, tx, w, method="logistic")]
    losses_ts = [compute_loss(y_t, tx_t, w, method="logistic")]
    
    pred = predict_labels_logistic(w, tx, threshold)
    acc_tr = [pred_accuracy(pred, y)]
    
    pred_t = predict_labels_logistic(w, tx_t, threshold)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    diff = 10
    n_iter=0
    nb_ES = 0
    velocity = 0
    
    while (n_iter < max_iters) and (patience > nb_ES):
        # compute gradient
        gd = compute_gradient(y, tx, w, lam, method="logistic")
        hess = compute_hessian(y, tx, w, lam)
        
        # compute next w
        velocity = momentum*velocity - gamma*np.linalg.lstsq(hess,gd)[0]
        w = w + velocity
        
        # compute loss and diff
        losses_tr.append(compute_loss(y, tx, w, method="logistic"))
        pred = predict_labels_logistic(w, tx, threshold)
        acc_tr.append(pred_accuracy(pred, y))
        
        losses_ts.append(compute_loss(y_t, tx_t, w, method="logistic"))
        pred_t = predict_labels_logistic(w, tx_t, threshold)
        acc_ts.append(pred_accuracy(pred_t, y_t))
        
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


def Grid_Search_logistic(y, tx, y_t, tx_t, initial_w, gamma=0.05, lam=0.1, max_iters = 100, momentum = 0, tol=1e-5, patience = 5):
    """ 
    Simpified version of logistic regressiong for faster computing of grid search
    
    """
    # Define parameters to store w
    w = initial_w
    
    pred_t = predict_labels_logistic(w, tx_t, 0.5)
    acc_ts = [pred_accuracy(pred_t, y_t)]
    
    velocity = 0
    
    diff = 10
    n_iter=0
    nb_ES = 0
    
    while (n_iter < max_iters) and (patience > nb_ES):
        # compute gradient
        gd = compute_gradient(y, tx, w, method="logistic")
        hess = compute_hessian(y, tx, w, lam)
        
        # compute next w
        velocity = momentum*velocity - gamma*np.linalg.solve(hess,gd)
        w = w + velocity
        
        pred_t = predict_labels_logistic(w, tx_t, 0.5)
        acc_ts.append(pred_accuracy(pred_t, y_t))

        n_iter = n_iter + 1
        if (abs(acc_ts[-1]-acc_ts[-2]) < tol):
                nb_ES = nb_ES + 1
        else:
            nb_ES = 0
    
    return max(acc_ts)


