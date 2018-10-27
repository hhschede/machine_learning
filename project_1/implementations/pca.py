import numpy as np

def PCA(data, threshold = 1):
    # computing the covariance matrix
    cov_matrix = np.cov(data.T)
    # computing the eigen-values and eigen-vectors
    eigenVal, eigenVec = np.linalg.eig(cov_matrix)
    
    # sorting the eigen-values and eigen-vectors
    idx = eigenVal.argsort()[::-1]   
    eigenVal = np.asarray(eigenVal[idx])
    eigenVec = np.asarray(eigenVec[:,idx])
    
    # keeping the eigen-vector that explains threshold% of the variance
    
    eigenVal=eigenVal/sum(eigenVal)

    sumEigenVal = [0]
    k = 0
    while((sumEigenVal[-1] < threshold) and (k < data.shape[1])):
        sumEigenVal.append(sumEigenVal[-1] + eigenVal[k])
        k = k+1

    #keep only kth first dimension

    eigenVec=eigenVec[:,:(k)]
    
    return eigenVal,eigenVec,sumEigenVal