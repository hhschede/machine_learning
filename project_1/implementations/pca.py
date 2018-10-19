import numpy as np

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