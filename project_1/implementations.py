import numpy as np
from helpers import *


# -----------------------------------------------------------------------------------

def compute_loss(y, tx, w, method = "MSE"):
    """Calculate the loss. using mse or mae.
    """
    if method == "MSE":
        err = y - np.dot(tx, w)
        loss = 1/(2*len(y))*np.sum(err**2)
    if method == "MAE":
        raise NotImplementedError
        
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

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma,write = False):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, 32):
        
            gd = compute_gradient(mini_y, mini_tx, w)
            loss = compute_loss(mini_y, mini_tx, w)
            w = w - gamma*gd
    # store w and loss
    ws.append(w)
    losses.append(loss)
    if write == True:
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


# -----------------------------------------------------------------------------------




class LogisticRegression:
    def __init__(self, learning=0.01, iterations=10000, fit_int=True, writing=False):
        self.learning = learning
        self.iterations = iterations
        self.fit_int = fit_int
        self.theta = 0
        self.writing = writing
        self.accuracy = 0
        self.loss = 0

    def add_int(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_func(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y, method = "gd"):
        X = self.concatenate_order_3(X)
    
        if self.fit_int:
            #adds the column of ones for w0 multiplication
            X = self.add_int(X)

        self.theta = np.ones(X.shape[1])
        self.accuracy = np.zeros(self.iterations)
        self.loss = np.zeros(self.iterations)
        
        if method == "gd":
            for i in range(self.iterations):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                grad = np.dot(X.T, (h - y)) / y.shape[0]

                self.theta -= self.learning * grad

                self.accuracy[i] = self.compute_accuracy(h,y)
                self.loss[i] = self.cost_func(h, y)

                if self.writing:
                    if i % 500 == 0:
                        print("iteration: {iter} | accuracy : {a}, loss : {l}".format(
                            iter = i, a = self.accuracy[i], l=self.loss[i] ))
        if method == "sgd":
            for i in range(self.iterations):   
                for mini_y, mini_X in batch_iter(y, X, 32):                
                    z = np.dot(mini_X, self.theta)
                    h = self.sigmoid(z)
                    grad = np.dot(mini_X.T, (h - mini_y)) / mini_y.shape[0]
                    self.theta -= self.learning * grad
                    
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                self.accuracy[i] = self.compute_accuracy(h,y)
                self.loss[i] = self.cost_func(h, y)

                if self.writing:
                    if i % 500 == 0:
                        print("iteration: {iter} | accuracy : {a}, loss : {l}".format(
                            iter = i, a = self.accuracy[i], l=self.loss[i] ))
                
                
            
        return self.accuracy, self.loss
       
    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def predict_class(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
    
    def compute_accuracy(self, h, y, threshold = 0.5):
        # faster way to compute the accuracy
        prediction = (h >= threshold)
        trues = y.astype(int) - prediction
        count = np.count_nonzero(trues==0)
        percent = count/len(trues)
        return percent
                          
    def concatenate_order_3(self, X):
        concatenate_X = []
        for col in X.T:
            col_squared = [entry**2 for entry in col]
            col_cubed = [entry ** 3 for entry in col]
            concatenate_X.append(col_squared)
            concatenate_X.append(col_cubed)
        concatenate_X = np.array(concatenate_X).T
        return (np.concatenate((X, concatenate_X), axis=1))
                   
     
        