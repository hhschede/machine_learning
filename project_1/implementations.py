import numpy as np
from helpers import *

def prepare_standardplot(title, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plotCurves(tr_loss, tr_acc, ts_loss, ts_acc, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(tr_loss, label = "training")
    ax1.plot(ts_loss, label = "validation")
    ax2.plot(tr_acc, label = "training")
    ax2.plot(ts_acc, label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig


def predict_labels_logistic(weights, data, threshold = 0.5):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    
    return y_pred

# -----------------------------------------------------------------------------------

def compute_loss(y, tx, w, lam = 0.1, method = "MSE"):
    """Calculate the loss."""

    if method == "MSE":
        err = y - tx.dot(w)
        loss = 1/2*np.mean(err**2)
    elif method == "MAE":  
        err = y - tx.dot(w)
        loss = np.mean(np.abs(err))
        
    elif method == "logistic":
        z = y * tx.dot(w)
        idx = z > 0
        loss = np.zeros(z.shape)
        loss[idx] = np.log(1 + np.exp(-z[idx]))
        loss[~idx] = (-z[~idx] + np.log(1 + np.exp(z[~idx])))
        loss = loss.sum() + 0.5 * lam * w.dot(w)
    elif method == "logistic2":
        h = tx.dot(w)
        loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    elif method == 'reg_logistic':
        loss = np.mean((-y * np.log(sigmoid(tx.dot(w))) - (1 - y) * np.log(1 - sigmoid(tx.dot(w)))) + 0.5 * lam * np.linalg.norm(w)) ** 2
    
    return loss

# -----------------------------------------------------------------------------------

def compute_gradient(y, tx, w, lam=0.1, method = "MSE"):
    """Compute the gradient."""
    
    err = y - tx.dot(w)
    if method == "MSE":
        grad = -tx.T.dot(err)/len(err)
        
    elif method == "MAE":
        grad = -tx.T.dot(np.sign(e))/len(y)
        
    elif method == "logistic":
        z = sigmoid(y * tx.dot(w))
        z0 = (z - 1) * y
        grad = tx.T.dot(z0) + lam * w
    elif method == "logistic2":
        h = sigmoid(tx.dot(w))
        grad = (tx.T.dot(h-y) + lam*w)/y.shape[0]
    
    return grad

# -----------------------------------------------------------------------------------

def compute_hessian(y, tx, w):
    """return the hessian of the loss function."""
    z = sigmoid(y * tx.dot(w))
    d = z*(1-z)
    wa = d[:, np.newaxis]*tx
    Hs = tx.T.dot(wa)
    hess = Hs 
    return hess

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
    
    idx = z > 0
    sig = np.zeros(z.shape)
    sig[idx] = 1. / (1 + np.exp(-z[idx]))
    sig[~idx] = np.exp(z[~idx]) / (1. + np.exp(z[~idx]))
    return sig

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

def logistic_regression(y, tx, initial_w, max_iters = 10000, gamma = 0.0005, method = "sgd",batch_size = 1000, writing = False):
    
    ws = np.zeros([max_iters + 1, tx.shape[1]])
    ws[0] = initial_w 
    losses_tr = []
    w = initial_w
    
    if method == "gd":
        for i in range(max_iters):
            grad = compute_gradient(y, tx, initial_w, lam=0, method = "logistic2")
            
            w -= gamma * grad
            ws[i+1] = w
            losses_tr.append(compute_loss(y, tx, w, lam = 0, method = 'logistic'))
            
            
            if writing:
                if i % 500 == 0:
                    print("iteration: {iter} | loss : {l}".format(
                        iter = i, l=losses_tr[-1] ))
    if method == "sgd":
        for i in range(max_iters):   
            for mini_y, mini_X in batch_iter(y, tx, batch_size):                
                grad = compute_gradient(y, tx, initial_w,lam=0, method = "logistic2")


                w -= gamma * grad
                
                ws[i+1] = w
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = 0, method = "logistic"))
                

        if writing:
            if i % 500 == 0:
                print("iteration: {iter} | loss : {l}".format(
                    iter = i, l=losses_tr[-1] ))
    return losses_tr, ws

# -----------------------------------------------------------------------------------

def logistic_hessian(y, tx, initial_w, max_iters = 100, tol=1e-8, writing = True):
    
    # Define parameters to store w and loss
    ws = [initial_w]
    w = initial_w
    losses = [compute_loss(y, tx, w, method="logistic")]
    diff = 10
    n_iter=0
    
    while (n_iter < max_iters) and (diff > tol):
        # compute gradient
        gd = compute_gradient(y, tx, w, 0.1, method="logistic")
        hess = compute_hessian(y, tx, w)
        
        # compute next w
        w = w - np.linalg.solve(hess,gd)
        
        # compute loss and diff
        loss = compute_loss(y, tx, w) 
        diff = abs(loss-losses[-1])
        
        # store w and loss and increment
        ws.append(w)
        losses.append(loss)
        n_iter += 1
        
        if writing == True:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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



def PCA(data, threshold = 1):
    cov_matrix = np.cov(data.T)
    eigenVal, eigenVec = np.linalg.eig(cov_matrix)

    idx = eigenVal.argsort()[::-1]   
    eigenVal = np.asarray(eigenVal[idx])
    eigenVec = np.asarray(eigenVec[:,idx])


    eigenVal=eigenVal/sum(eigenVal)


    sumEigenVal = [0]
    k = 0
    while(sumEigenVal[-1] < threshold):
        sumEigenVal.append(sumEigenVal[-1] + eigenVal[k])
        k = k+1

    #keep only kth first dimension

    eigenVec=eigenVec[:,:(k)]
    
    return eigenVal,eigenVec,sumEigenVal