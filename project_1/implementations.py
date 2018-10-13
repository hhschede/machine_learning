import numpy as np
from helpers import *

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
        loss = np.mean((-y * np.log(sigmoid(tx.dot(w))) - (1 - y) * np.log(1 - sigmoid(tx.dot(w)))))
    
    elif method == 'reg_logistic':
        loss = np.mean((-y * np.log(sigmoid(tx.dot(w))) - (1 - y) * np.log(1 - sigmoid(tx.dot(w)))) + 0.5 * lam * np.linalg.norm(w)) ** 2
    
    return loss

# -----------------------------------------------------------------------------------

def compute_gradient(y, tx, w, method = "MSE"):
    """Compute the gradient."""
    
    err = y - tx.dot(w)
    if method == "MSE":
        grad = -tx.T.dot(err)/len(err)
        
    elif method == "MAE":
        grad = -tx.T.dot(np.sign(e))/len(y)
    return grad

# -----------------------------------------------------------------------------------

def least_squares_GD(y, tx, initial_w, tol = 1e-8, max_iters = 10000, gamma = 0.05, write = False):
    """Gradient descent algorithm."""
    
    # Define parameters to store w and loss
    ws = [initial_w]
    w = initial_w
    losses = [compute_loss(y, tx, w)]
    diff = losses[0]
    n_iter=0
    
    while (n_iter < max_iters) and (diff > tol):
        # compute gradient
        gd = compute_gradient(y, tx, w)
        
        # compute next w
        w = w - gamma*gd
        
        # compute loss and diff
        loss = compute_loss(y, tx, w) 
        diff = abs(loss-losses[-1])
        
        # store w and loss and increment
        ws.append(w)
        losses.append(loss)
        n_iter += 1
        
        if write == True:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

# -----------------------------------------------------------------------------------

def least_squares_SGD(y, tx, initial_w, batch_size = 1000, tol = 1e-4,patience = 1, max_iters = 1000, gamma = 0.05, write = False):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    w = initial_w
    losses = [compute_loss(y, tx, w)]
    diff = losses[0]
    n_iter=0
    nb_ES = 0
    
    
    while (n_iter <= max_iters) and (nb_ES < patience): 
        for mini_y, mini_tx in batch_iter(y, tx, batch_size):
            # compute gradient
            gd = compute_gradient(mini_y, mini_tx, w)
            
            # compute next w
            w = w - gamma*gd
            
            # compute loss
            loss = compute_loss(y, tx, w)
            
            diff = abs(loss-losses[-1])
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
            n_iter += 1
            if (diff < tol):
                nb_ES = nb_ES + 1
            
        if write and (n_iter % 500 == 0):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))
    return losses, ws

# -----------------------------------------------------------------------------------

def least_squares(y, tx):
    """Calculate the least squares solution."""
    
    X = tx.T.dot(tx)
    Y = tx.T.dot(y)
    
    return np.linalg.solve(X, Y)

# -----------------------------------------------------------------------------------
# this will have to be changed to the build_poly(x, degree): function!!!!!!!!

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x[:,1:], deg)]
    return poly


# -----------------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Calculates ridge regression."""
    
    xTx = tx.T.dot(tx)
    X = xTx + (2 * len(y) * lambda_ * np.identity(len(xTx)))
    Y = tx.T.dot(y)
    return np.linalg.solve(X, Y)

# -----------------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-abs(z)))

# -----------------------------------------------------------------------------------

def compute_accuracy_logistic_regression(y, tx, w, threshold = 0.5):
    # faster way to compute the accuracy
    h = sigmoid(np.dot(tx, w))
    prediction = (h >= threshold)
    trues = y.astype(int) - prediction
    count = np.count_nonzero(trues==0)
    percent = count/len(trues)
    return percent

# -----------------------------------------------------------------------------------

def logistic_regression(y, tx, initial_w, max_iters = 10000, gamma = 0.01, method = "sgd",batch_size = 250, writing = False):
    
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
            losses.append(compute_loss(y, tx, w, method = "logistic"))

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
                losses.append(compute_loss(mini_y, mini_X, w, method = "logistic"))

        if writing:
            if i % 500 == 0:
                print("iteration: {iter} | loss : {l}".format(
                    iter = i, l=losses[-1] ))
    return losses, ws
       
# --------------------------------------------------------------------------------

def reg_logistic_regression(y, tx, initial_w, lamb, max_iters = 10000, gamma = 0.01, methods = "sgd", batch_size = 250, writing = False):
    
    ws = np.zeros([max_iters + 1, tx.shape[1]])
    ws[0] = initial_w 
    losses = []
    w = initial_w
    
    if methods == "gd":
        for i in range(max_iters):
            h = sigmoid(np.dot(tx, w))
            grad = np.dot(tx.T, (h - y)) / y.shape[0] + lamb * np.linalg.norm(w) / y.shape[0]
            w -= gamma * grad
            ws[i+1] = w
            losses.append(compute_loss(y, tx, w, lam = lamb, method = 'reg_logistic'))

            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss : {l}".format(
                        iter = i, l=losses[-1] ))

                    

    if methods == "sgd":
        for i in range(max_iters):   
            for mini_y, mini_X in batch_iter(y, tx, batch_size):                
                h = sigmoid(np.dot(mini_X, w))
                grad = np.dot(mini_X.T, (h - mini_y)) / mini_y.shape[0] + lamb * np.linalg.norm(w) / y.shape[0]
                w -= gamma * grad
                ws[i+1] = w
                losses.append(compute_loss(mini_y, mini_X, w, lam = lamb, method = 'reg_logistic'))

            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss : {l}".format(
                        iter = i, l=losses[-1] ))

    return losses, ws
                          
    
                   
     
        
