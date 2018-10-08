import numpy as np
from helpers import *

# -----------------------------------------------------------------------------------

def compute_loss(y, tx, w, method = "MSE"):
    """Calculate the loss."""
    
    err = y - tx.dot(w)
    if method == "MSE":
        loss = 1/2*np.mean(err**2)
    elif method == "MAE":  
        loss = np.mean(np.abs(err))
    elif method == "logistic":
        loss = np.mean((-y * np.log(sigmoid(tx.dot(w))) - (1 - y) * np.log(1 - sigmoid(tx.dot(w)))))
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

def least_squares_SGD(y, tx, initial_w, batch_size = 1000, tol = 1e-4, max_iters = 1000, gamma = 0.05, write = False):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    w = initial_w
    losses = [compute_loss(y, tx, w)]
    diff = losses[0]
    n_iter=0
    
    while (n_iter < max_iters) and (diff > tol): 
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
    
    x_poly = np.ones((len(x),1))
    for d in range(degree):
        x_poly = np.c_[x_poly, np.power(x, d+1)]
    return x_poly

# -----------------------------------------------------------------------------------

def polynomial_regression(y, tx, degree):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    
    # Form dataset to do polynomial regression.
    x_poly = build_poly(tx, degree)

    # Least squares using normal equations
    w = least_squares(y, tx)
    
    return w

def ridge_regression(y, tx, lambda_):
    """Calculates ridge regression."""
    
    xTx = tx.T.dot(tx)
    X = xTx + (2 * len(y) * lambda_ * np.identity(len(xTx)))
    Y = tx.T.dot(y)
    return np.linalg.solve(X, Y)

# -----------------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
       

                          
    
                   
     
        
