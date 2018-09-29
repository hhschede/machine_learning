# file containing the logistic regression function

import numpy as np

class LogisticRegression:
    def __init__(self, learning=0.01, iterations=10000, fit_int=True, writing=False):
        self.learning = learning
        self.iterations = iterations
        self.fit_int = fit_int
        self.theta = 0
        self.writing = writing

    def add_int(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_func(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_int:
            X = self.add_int(X)

        self.theta = np.ones(X.shape[1])

        for i in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            grad = np.dot(X.T, (h - y)) / y.shape[0]
            self.theta -= self.learning * grad
            #print(self.theta)
            if self.writing:
                if i % 100 == 0:
                    z = np.dot(X, self.theta)
                    h = self.sigmoid(z)
                    print(self.cost_func(h, y))

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def predict_class(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold