import numpy as np
from helpers import *


# -----------------------------------------------------------------------------------

def compute_loss(y, tx, w, method = "MAE"):
    """Calculate the loss. using mse or mae.
    """
    
    if method == "MSE":
        err = y - tx.dot(w)
        loss = 1/2*np.mean(err**)
    elif method == "MAE":  
        err = y - tx.dot(w)
        loss = np.mean(np.abs(err))
    elif method == "logistic":
        try: 
            loss = np.mean((-y * np.log(tx) - (1 - y) * np.log(1 - tx)))
        except RuntimeWarning:
            pass
    return loss

# -----------------------------------------------------------------------------------

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - np.dot(tx, w)
    g = -1/len(y) * tx.T.dot(err)
    return g

# -----------------------------------------------------------------------------------

def gradient_descent(y, tx, initial_w, max_iters, gamma, write = False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # TODO: compute gradient and loss
        gd = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)  
        
        # TODO: update w by gradient
        w = w - gamma*gd
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        if write == True:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

# -----------------------------------------------------------------------------------

def stochastic_gradient_descent(y, tx, initial_w, batch_size = 500, max_iters = 10000, gamma = 0.05,write = False):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, 32):
            gd = compute_gradient(mini_y, mini_tx, w)
            w = w - gamma*gd
            # store w and loss
            ws.append(w)
            losses.append(compute_loss(y, tx, w,method = "MSE"))
        if write and (n_iter % 500 == 0):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))
    return losses, ws


# -----------------------------------------------------------------------------------

def prepare_data_order_3(X):
    
    concatenate_X = []
    for col in X.T:
        col_squared = [entry**2 for entry in col]
        col_cubed = [entry ** 3 for entry in col]
        concatenate_X.append(col_squared)
        concatenate_X.append(col_cubed)
    concatenate_X = np.array(concatenate_X).T
    data = (np.concatenate((X, concatenate_X), axis=1))
    
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, data), axis=1)

def prepare_data_order_2(X):
    
    concatenate_X = []
    
    
    for col in X.T:
        col_squared = [entry**2 for entry in col]
        concatenate_X.append(col_squared)
    concatenate_X = np.array(concatenate_X).T
    data = (np.concatenate((X, concatenate_X), axis=1))
    
    intercept = np.ones((X.shape[0], 1))
    
    return np.concatenate((intercept, data), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

    

def compute_accuracy_logistic_regression(y, tx, w, threshold = 0.5):
    # faster way to compute the accuracy
    h = sigmoid(np.dot(tx, w))
    prediction = (h >= threshold)
    trues = y.astype(int) - prediction
    count = np.count_nonzero(trues==0)
    percent = count/len(trues)
    return percent



def logistic_regression(y, tx, initial_w, max_iters = 10000, gamma = 0.01, method = "sgd",batch_size = 250 writing = False):
    
    ws = np.zeros([max_iters + 1, tx.shape[1]])
    ws[0] = initial_w 
    losses = []
    w = initial_w
    
    if method == "gd":
        for i in range(max_iters):
            h = sigmoid(np.dot(tx, w))
            grad = np.dot(tx.T, (h - y)) / y.shape[0]

            w -= gamma * grad
            ws[i+1] = w
            losses.append(compute_loss(y, h, w, method = "logistic"))

            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss : {l}".format(
                        iter = i, l=losses[-1] ))
    if method == "sgd":
        for i in range(max_iters):   
            for mini_y, mini_X in batch_iter(y, tx, batch_size):                
                h = sigmoid(np.dot(mini_X, w))
                grad = np.dot(mini_X.T, (h - mini_y)) / mini_y.shape[0]
                
                w -= gamma * grad
                
                ws[i+1] = w
                losses.append(compute_loss(mini_y, h, w, method = "logistic"))

        if writing:
            if i % 500 == 0:
                print("iteration: {iter} | loss : {l}".format(
                    iter = i, l=losses[-1] ))
    return losses, ws
       

                          
    
                   
     
        