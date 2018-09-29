import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def test_train(X, y, test_ratio=0.3):
    """The test_train function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    the input X is a numpy matrix with samples in rows and features in columns.
    the input y is a numpy array with entry corresponding to the target value
    of the corresponding sample in X. This may be a class or a float.
    The test_ratio is a float, default 0.3, that sets the test set fraction of
    the entire dataset to 0.3 and keeps the other part for training"""

    import random
    import numpy as np

    # Perform shuffling
    idx_shuffled = np.random.permutation(np.arange(0, X.shape[0]))
    # return shuffled X and y
    X_shuff = []
    y_shuff = []
    for sample in idx_shuffled:
        X_shuff.append(X[sample])
        y_shuff.append(y[sample])
    X_shuff = np.matrix(X_shuff)
    y_shuff = np.array(y_shuff)

    # Cut the data set into train and test
    test_num = int(X.shape[0] * test_ratio)
    train_num = int(X.shape[0] - test_num)

    # print(X_shuff)
    X_train = X_shuff[0:train_num]
    y_train = y_shuff[0:train_num]
    X_test = X_shuff[train_num:]
    y_test = y_shuff[train_num:]

    return X_train, y_train, X_test, y_test


def pred_accuracy(predict, y):
    trues = []
    for num, val in enumerate(y):
        if int(val) == predict[num]:
            trues.append(1)
        else:
            trues.append(0)
    trues = np.array(trues)
    percent = trues.mean()
    print(percent)
    return percent


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


(labels, data, ids) = load_csv_data('/Users/halimaschede/Desktop/PycharmProjects/cs433/train.csv')  # load data



# CLEAN UP DATA

# set values of -999 to NaN. Then calculate the means of the features. Replace NaN values with new values.
data_process = np.array(data[:1000,:])
labels_select = np.array(labels[:1000])
lab = []
for entry in labels_select:
    if int(entry) == 1:
        lab.append(1)
    else:
        lab.append(0)
lab = np.array(lab)
data_process[data_process == -999] = np.nan
# print(data_process.shape)

# filter out really shitty features
# retrieve percentage for each feature - how many nan's are there?

nan_count = []
for c in data_process.T:
    count = 0
    for e in c:
        if np.isnan(e):
            count += 1
    pcent = count / data_process.shape[0]
    nan_count.append(pcent)

# filter out features which have nan values of 50%

data_set_filtered = []
for idx, entry in enumerate(nan_count):
    if entry < 0.6:
        data_set_filtered.append(data_process.T[idx]) #append the column of the original dataset that is good
data_set_filtered = np.array(data_set_filtered).T #save that shit as an np array

# filter out samples that have more than half of their features as nan (crappy samples)

nan_count_2 = []
data_set_filtered_2 = [] # dataset filtered for columns and samples
y = [] # array that gets rid of entries that are no longer corresponding in the dataframe
for sample in data_set_filtered:
    count = 0
    for col in sample:
        if np.isnan(col):
            count += 1
    pcent = count / data_set_filtered.shape[1]
    nan_count_2.append(pcent)

for idx, entry in enumerate(nan_count_2):
    if entry < 0.15:
        y.append(lab[idx])
        data_set_filtered_2.append(data_set_filtered[idx])
data_set_filtered_2 = np.array(data_set_filtered_2) # turn dat shit into an array
y = np.array(y) # also this one gotta be an array

# print new dimensions of the dataframe after filtering

print('The original dimensions of the training data set was {0} samples'
      ' and {1} columns. After feature and sample filtering, there are'
      ' {2} samples and {3} columns'.format(data_process.shape[0],
                                            data_process.shape[1],
                                            data_set_filtered_2.shape[0],
                                            data_set_filtered_2.shape[1]))

data_nan = data_set_filtered_2.copy() # variable reassigned
column_means = [] # create list with average values of columns, excluding nans
for column in data_nan.T:
    column_means.append(np.nanmean(column))

inds = np.where(np.isnan(data_nan)) # variable containing locations of nan in data frame
data_nan[inds] = np.take(column_means, inds[1]) # reassign locations of nan to the column means


# standardize and normalize the features
column_variance = []
for idx, column in enumerate(data_nan.T):
    mean = np.mean(column)
    variance = np.var(column)
    for entry, value in enumerate(column):
        data_nan[entry, idx] = (value - mean)/variance # scale dat shit

# COMPRESS DATA BY PCA. Note that this is an unsupervised method and might not be a good alternative

# pca_ = pca(percent_rep=0.9)
# pca_.fit(data_nan)
# pca_data = pca_.transform(data_nan)
#
# pca_data_A = []
# pca_data_B = []
# for idx, entry in enumerate(labels_select):
#     if entry == 1:
#         pca_data_A.append(pca_data[idx])
#     else:
#         pca_data_B.append(pca_data[idx])
# pca_data_A = np.array(pca_data_A)
# pca_data_B = np.array(pca_data_B)
#
# fig = plt.figure()
# scattered = plt.axes(projection = '3d')
# scattered.scatter3D(pca_data_A[:,0], pca_data_A[:,1], pca_data_A[:,2], c='blue')
# scattered.scatter3D(pca_data_B[:,0], pca_data_B[:,1], pca_data_B[:,2], c='red')
#
# plt.show()
# pca_int = logreg.add_int(pca_data)
# prediction = logreg.predict_class(pca_int)
# print(prediction)

# add some non-linear variants of each of the features that are included in the data set
# do this by constructing a new array, which will then be concatenated with the old one
# note that the addition of the squared features increases accuracy by a few percentage
# but there is a smaller, but clear, difference with the cubed features

concatenate_me = []

for col in data_nan.T:
    col_squared = [entry**2 for entry in col]
    col_cubed = [entry ** 3 for entry in col]
    concatenate_me.append(col_squared)
    concatenate_me.append(col_cubed)

concatenate_me = np.array(concatenate_me).T
data_concatenated = np.concatenate((data_nan, concatenate_me), axis=1)


# perform the linear regression
# i just noticed that the prediction accuracy also depends quite a bit on the initialized w.
# for example if you initialize them all to ones instead of zeros, accuracy increases. overfitting?

logreg = LogisticRegression()
logreg.fit(data_concatenated, y)
data_con_int = logreg.add_int(data_concatenated)
prediction = logreg.predict_class(data_con_int)
prediction = np.array(prediction)


print(pred_accuracy(prediction, y))


# SPLIT DATA

#(X_train, y_train, X_test, y_test) = test_train(data, labels, test_ratio=0.3)  # split into train and test set


np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

# # CALCULATE FISHER SCORE VARIANT
#
# scores = []
# means_A = []
# var_A = []
# means_B = []
# var_B = []
# class_A = np.array(data_nan[np.argwhere(y == 1)])# separate out classes
# class_A = class_A[:,0,:]
# class_B = np.array(data_nan[np.argwhere(y == 0)])
# class_B = class_B[:,0,:]
#
# for c in class_A.T:
#     means_A.append(np.mean(c))
#     var_A.append(np.var(c))
# for c in class_B.T:
#     means_B.append(np.mean(c))
#     var_B.append(np.mean(c))
# for x in range(len(means_A)):
#     scores.append(means_A[x] - means_B[x]) # score function
# print(np.array(scores))
#
#
# # fig = plt.figure()
# # scattered = plt.axes(projection = '3d')
# # scattered.scatter3D(class_A[:,12], class_A[:,28], class_A[:,4], c='blue')
# # scattered.scatter3D(class_B[:,12], class_B[:,28], class_B[:,4], c='red')
# fig = plt.figure()
# plt.scatter(class_B[:,12], class_B[:,10], c='blue', alpha=0.2)
# plt.scatter(class_A[:,12], class_A[:,10], c='red', alpha=0.2)
#
# plt.show()
