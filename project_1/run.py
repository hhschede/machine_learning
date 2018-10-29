
# coding: utf-8

# Importation of modules and functions
# ===

# In[ ]:


# Modules
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import datetime
import random
import warnings

# Functions
sys.path.insert(0, './implementations/')
from implementations import *
from preprocessing import *
from pca import *
from plot import *
from helpers import *

# Autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Set random seed
np.random.seed(1)


# Training data loading
# ===

# In[ ]:


(labels_raw, data_raw, ids_raw) = load_csv_data("data/train.csv")


# Testing data loading
# ===

# In[ ]:


(labels_t, data_raw_t, ids_t) = load_csv_data("data/test.csv")


# Get jet indexes
# ===

# In[ ]:


# Get feature jet_num
jets = data_raw[:,22]
jets_t = data_raw_t[:,22]

# Get index of samples with appropriate jet
idx_jet0 = np.argwhere(jets == 0)[:,0]
idx_jet1 = np.argwhere(jets == 1)[:,0]
idx_jet2 = np.argwhere(jets >= 2)[:,0]

idx_jet0_t = np.argwhere(jets_t == 0)[:,0]
idx_jet1_t = np.argwhere(jets_t == 1)[:,0]
idx_jet2_t = np.argwhere(jets_t >= 2)[:,0]


# Separate data relative to jets
# ===

# In[ ]:


# Remove jet_num feature as it will be constant after separation
data_raw = np.delete(data_raw, 22, axis=1)
data_raw_t = np.delete(data_raw_t, 22, axis=1)

# Split data relative to jets
data_tr_j0 = data_raw[idx_jet0,:]
data_tr_j1 = data_raw[idx_jet1,:]
data_tr_j2 = data_raw[idx_jet2,:]

data_ts_j0 = data_raw_t[idx_jet0_t,:]
data_ts_j1 = data_raw_t[idx_jet1_t,:]
data_ts_j2 = data_raw_t[idx_jet2_t,:]

# Split labels relative to jets
lab_j0 = labels_raw[idx_jet0]
lab_j1 = labels_raw[idx_jet1]
lab_j2 = labels_raw[idx_jet2]

lab_j0_t = labels_t[idx_jet0_t]
lab_j1_t = labels_t[idx_jet1_t]
lab_j2_t = labels_t[idx_jet2_t]


# Data filtering and transformation
# ===

# In[ ]:


# Filtering features, missing values and outliers
data_j0, data_j0_t = process_data(data_tr_j0, data_ts_j0)
data_j1, data_j1_t = process_data(data_tr_j1, data_ts_j1)
data_j2, data_j2_t = process_data(data_tr_j2, data_ts_j2)


# In[ ]:


# Transforming data using polynomials, log and interaction terms
y_j0, tx_j0, y_j0_t, tx_j0_t = transform_data(data_j0, data_j0_t, lab_j0, lab_j0_t)
y_j1, tx_j1, y_j1_t, tx_j1_t = transform_data(data_j1, data_j1_t, lab_j1, lab_j1_t)
y_j2, tx_j2, y_j2_t, tx_j2_t = transform_data(data_j2, data_j2_t, lab_j2, lab_j2_t)


# Logistic regression using Newton's method
# ===

# In[ ]:


max_iter = 50

initial_w = np.zeros(tx_j0.shape[1])
losses, losses_t, acc, acc_t, w_0 = logistic_hessian(y_j0, tx_j0, y_j0_t, tx_j0_t, initial_w, gamma = 0.08, lam = 0.01, momentum = 0.1, max_iters = max_iter)

initial_w = np.zeros(tx_j1.shape[1])
losses, losses_t, acc, acc_t, w_1 = logistic_hessian(y_j1, tx_j1, y_j1_t, tx_j1_t, initial_w, gamma = 0.0575, lam = 10, momentum = 0, max_iters = max_iter)

initial_w = np.zeros(tx_j2.shape[1])
losses, losses_t, acc, acc_t, w_2 = logistic_hessian(y_j2, tx_j2, y_j2_t, tx_j2_t, initial_w, gamma = 0.065, lam = 10, momentum = 0, max_iters = max_iter)


# Kaggle submission
# ===

# In[ ]:


pred_t = np.zeros(ids_t.shape)

# Predict labels and recombine them into one vector for kaggle submission
pred_t[idx_jet0_t] = predict_labels_logistic(w_0, tx_j0_t, 0.5)
pred_t[idx_jet1_t] = predict_labels_logistic(w_1, tx_j1_t, 0.5)
pred_t[idx_jet2_t] = predict_labels_logistic(w_2, tx_j2_t, 0.5)

# Create submission
name = "final_submission.csv"
create_csv_submission(ids_t, pred_t, name)

