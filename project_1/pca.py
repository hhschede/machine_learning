# file containing the PCA functions

import numpy as np

class pca(object):
    """find principal components and order them by eigenvalue. comes with the two methods fit and transform"""
    def __init__(self, percent_rep):
        self.percent_rep = percent_rep # decimal value of the percent of the original data we want to represent
        self.eigenvectors = 0
        self.eigenvalues = 0

    def fit(self, x):
        import numpy as np
        # x is a dataset that is already standardized as a numpy matrix
        # create covariance matrix
        cov_matrix = np.cov(x.T)
        eigen, eigenvectors = np.linalg.eig(cov_matrix)

        eigen = eigen.astype(float)
        eigenvectors = eigenvectors.astype(float)
        eigen_sort = np.sort(eigen)
        eigenvector_sort = eigenvectors[np.argsort(eigen)]

        # calculate the number of eigenvalues needed to retrieve desired percent
        percent = 0
        selected_eigenvalues = []
        selected_eigenvectors = []

        for i in np.arange(1,x.shape[0]):
            percent += eigen_sort[-i]/eigen_sort.sum()
            selected_eigenvalues.append(eigen_sort[-i])
            selected_eigenvectors.append(eigenvector_sort[-i])
            if percent > self.percent_rep:
                break

        self.eigenvalues = np.array(selected_eigenvalues)
        self.eigenvectors = np.array(selected_eigenvectors)
        return self.eigenvalues, self.eigenvectors

    def transform(self, input):
        import numpy as np
        transformed = np.dot(self.eigenvectors, input.T).T
        return transformed