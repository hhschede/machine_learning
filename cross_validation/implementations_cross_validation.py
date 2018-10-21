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


def predict_labels_logistic(weights, data, threshold = 0.7):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred < threshold)] = -1
    y_pred[np.where(y_pred >= threshold)] = 1
    
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
        err = y - sigmoid(tx.dot(w))
        loss = 0.5*np.mean(err**2)
    
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
        grad = (tx.T.dot(z0) + lam * w)/len(y)
    return grad

# -----------------------------------------------------------------------------

def build_model_data(data, labels):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(labels)
    tx = np.c_[np.ones(num_samples), data]
    return labels, tx

# -----------------------------------------------------------------------------


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = x
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

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
def build_interact_terms(x):
    """Calculates interaction terms for each feature."""
    
    interact = (x[:,1:].T * x[:,0]).T
    for i in range(1, x.shape[1]):
        interact = np.c_[interact, (x[:,(i+1):].T*x[:,i]).T]
    return interact

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------------

def compute_hessian(y, tx, w, lam):
    """return the hessian of the loss function."""
    z = sigmoid(tx.dot(w))
    D = z * (1-z)
    XD = tx * D.reshape(-1,1)
    hess = tx.T.dot(XD) + lam*np.eye((tx.shape[1]))
    return hess/len(y)
# -----------------------------------------------------------------------------------



def cross_val(y, x, k):
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
        
        # standardize the sets
        


        x_train, mean, variance = standardize(x_tr)
        x_test = standardize_test(x_te, mean, variance)
        
        eigVal, eigVec, sumEigVal = PCA(x_train, threshold = 0.90)
        x_train = x_train.dot(eigVec)
        x_test = x_test.dot(eigVec)
        
        y_tr, x_train = build_model_data(x_train, y_tr)
        y_te, x_test = build_model_data(x_test, y_te)
        
        

        
        yield np.array(y_tr), np.array(x_train), np.array(y_te), np.array(x_test) #this is a generator! call next(object) for next set

        
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# LEARNING ALGORITHM FUNCTIONS
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def least_squares_GD(y, tx, initial_w, tol = 1e-5, max_iters = 10000, gamma = 0.05, k=4,  write = False):
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
        n_iter = 0
        w = initial_w
        while n_iter < max_iters:
            # compute gradient
            gd = compute_gradient(y, tx, w)
            # compute next w
            w = w - gamma*gd
            n_iter += 1
            
        return w
    
    # This else is for determining if this is a good way to select the model - return test and training errors
    else:
        gen = cross_val(y, tx, k) # initiate generator object
        w_final = []
        accuracies = []
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen)
            w = np.random.rand(x_tr.shape[1])
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

def least_squares_SGD(y, tx, initial_w, batch_size = 1000, tol = 1e-4,patience = 1, max_iters = 1000, gamma = 0.05, k=4, write = False):
    """Stochastic gradient descent algorithm."""
    # test_loss = []
    # training_loss = []
    w = initial_w
    # losses = [compute_loss(y, tx, w)]
    # diff = losses[0]
    n_iter=0
    nb_ES = 0
    
    if k == 0:
        
        while n_iter <= max_iters: 
            for mini_y, mini_tx in batch_iter(y, tx, batch_size):
                # compute gradient
                gd = compute_gradient(mini_y, mini_tx, w)

                # compute next w
                w = w - gamma*gd

                # compute loss
                # loss = compute_loss(y, tx, w)

                # diff = abs(loss-losses[-1])

                # store w and loss
                # ws.append(w)
                # losses.append(loss)
                n_iter += 1
                #if (diff < tol):
                #    nb_ES = nb_ES + 1

            if write and (n_iter % 500 == 0):
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))
        return w
    
    else:

        gen = cross_val(y, tx, k) # initiate generator object
        w_final = []
        accuracies = []
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen)
            w = np.random.rand(x_tr.shape[1])
            # Define parameters to store w and loss
            # ws = [initial_w]
            w = initial_w
            # my_losses = [compute_loss(y_tr, x_tr, w)]
            # test_mylosses = []
            # loss = my_losses[0]
            n_iter=0
            while n_iter < max_iters:
                for mini_y, mini_tx in batch_iter(y_tr, x_tr, batch_size):
                    gd = compute_gradient(mini_y, mini_tx, w)
                    w = w - gamma*gd
                    # loss = compute_loss(y, tx, w)
                    # diff = abs(loss-losses[-1])
                    # losses.append(loss)
                    n_iter += 1
                    #if (diff < tol):
                    #    nb_ES = nb_ES + 1
                        
                if write and (n_iter % 500 == 0):
                    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                        bi=n_iter, ti=max_iters - 1, l=losses[-1], w0=w[0], w1=w[1]))
                    
            # Append list of test accuracies
            w_final.append(w)
            test_pred_lab = predict_labels(w, x_te)
            accuracies.append(pred_accuracy(test_pred_lab, y_te))
       
        return w_final, accuracies
# -----------------------------------------------------------------------------------

def least_squares(y, tx, k=4):
    """Calculate the direct least square solution with cross validation option
    
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
        w = np.linalg.solve(X,Y)
        return w

    # If k > 0, determine if this is a good way to select the model (retrieve variance and bias)
    else:
        w_final = []
        accuracies = []
        gen = cross_val(y, tx, k) # initiate generator object
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen) # take next subtraining and subtest sets
            X = x_tr.T.dot(x_tr)
            Y = x_tr.T.dot(y_tr)
            try:
                w = np.linalg.solve(X,Y)
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
# this will have to be changed to the build_poly(x, degree): function!!!!!!!!

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = x
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


# -----------------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_, k=4):
    """Calculates ridge regression."""
    
    if k==0:
        
        xTx = tx.T.dot(tx)
        X = xTx + (2 * len(y) * lambda_ * np.identity(len(xTx)))
        Y = tx.T.dot(y)
        return np.linalg.solve(X, Y)
    else:
        w_final = []
        accuracies = []
        gen = cross_val(y, tx, k) # initiate generator object
        for i in np.arange(k):
            y_tr, x_tr, y_te, x_te = next(gen) # take next subtraining and subtest sets
            xTx = x_tr.T.dot(x_tr)
            X = xTx + (2 * len(y_tr) * lambda_ * np.identity(len(xTx)))
            Y = x_tr.T.dot(y_tr)
            w = np.linalg.solve(X,Y)
            
            # Append test accuracies
            w_final.append(w)
            test_pred_lab = predict_labels(w, x_te)
            accuracies.append(pred_accuracy(test_pred_lab, y_te))
        return w_final, accuracies
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
            grad = compute_gradient(y, tx, initial_w, lam=0, method = "logistic")
            
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
                grad = compute_gradient(y, tx, initial_w,lam=0, method = "logistic")


                w -= gamma * grad
                
                ws[i+1] = w
                losses_tr.append(compute_loss(mini_y, mini_X, w,lam = 0, method = "logistic"))
                

        if writing:
            if i % 500 == 0:
                print("iteration: {iter} | loss : {l}".format(
                    iter = i, l=losses_tr[-1] ))
    return losses_tr, ws

# -----------------------------------------------------------------------------------

def logistic_hessian(y, tx, y_t, tx_t, initial_w, gamma=0.05, lam=0.1, max_iters = 100, tol=1e-8, patience = 1, writing = True, threshold = 0.5):
    
    # Define parameters to store w
    w = initial_w
    
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
    
    while (n_iter < max_iters) and (patience > nb_ES):
        # compute gradient
        gd = compute_gradient(y, tx, w, method="logistic")
        hess = compute_hessian(y, tx, w, lam)
        
        # compute next w
        w = w - gamma*np.linalg.solve(hess,gd)
        
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