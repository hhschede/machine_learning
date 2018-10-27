import numpy as np
from implementations import *
from pca import *

def process_data(data, data_t, feature_filtering = True, remove_outlier = True, replace = 'mean'):
    """The process_data function prepares the train and test data by performing
    some data cleansing techniques. Missing values which are set as -999
    are replaced by NaN, then the means of each features are calculated
    in order to replace NaN values by the means. Features and samples with
    too much missing values are deleted by this function. Returns filtered
    dataset and column idx to be removed as well as column means and variance
    (required for transformation of unseen data."""

    
    data_process_tr = np.array(data)
    data_process_ts = np.array(data_t)
    
    # Setting missing values (-999) as NaN
    data_process_tr[data_process_tr == -999] = np.nan
    data_process_ts[data_process_ts == -999] = np.nan

    # Filtering weak features and samples

    # Retrieving percentage for each feature 
    
    if feature_filtering:
        print("Filtering features")
        
        nan_count = np.count_nonzero(np.isnan(data_process_tr),axis=0)/data_process_tr.shape[0]
        
        idx_del = []
        
        for idx, entry in enumerate(nan_count):
            if (entry >= 1):
                idx_del.append(idx)
        data_process_tr = np.delete(data_process_tr, idx_del,1).copy()
        data_process_ts = np.delete(data_process_ts, idx_del,1).copy()
        
        

    if remove_outlier:
        
        print("Finding and replacing outliers by column mean")
        
        data_tr = np.zeros(data_process_tr.shape)
        data_ts = np.zeros(data_process_ts.shape)
        column_means = np.nanmean(data_process_tr, axis=0)
        column_means_t = np.nanmean(data_process_ts, axis=0)
        column_std = np.nanstd(data_process_tr, axis=0)
        column_std_t = np.nanstd(data_process_ts, axis=0)

        for i in range(len(column_means)):
            col = data_process_tr[:,i]
            thresh_sup = column_means[i] + 3*column_std[i]
            thresh_inf = column_means[i] - 3*column_std[i]

            col_t = data_process_ts[:,i]
            thresh_ts = column_means_t[i] + 3*column_std_t[i]
            thresh_ti = column_means_t[i] - 3*column_std_t[i]
            
            mean_wo_outlier = np.mean(col[ np.array([(e <= thresh_sup and e >= thresh_inf) if ~np.isnan(e) else False for e in col], dtype=bool) ])
            
            col[ np.array([e >= thresh_sup if ~np.isnan(e) else False for e in col], dtype=bool) ] = mean_wo_outlier
            col[ np.array([e <= thresh_inf if ~np.isnan(e) else False for e in col], dtype=bool) ] = mean_wo_outlier
            
            mean_wo_outlier_t = np.mean(col_t[ np.array([(e <= thresh_ts and e >= thresh_ti) if ~np.isnan(e) else False for e in col_t], dtype=bool) ])
            
            col_t[ np.array([e >= thresh_ts if ~np.isnan(e) else False for e in col_t], dtype=bool) ] = mean_wo_outlier_t
            col_t[ np.array([e <= thresh_ti if ~np.isnan(e) else False for e in col_t], dtype=bool) ] = mean_wo_outlier_t

            data_tr[:,i] = col

            data_ts[:,i] = col_t

        data_process_tr = data_tr.copy()
        data_process_ts = data_ts.copy()
        
    column_means = np.nanmean(data_process_tr, axis=0)
    column_means_t = np.nanmean(data_process_ts, axis=0)
    column_std = np.nanstd(data_process_tr, axis=0)
    column_std_t = np.nanstd(data_process_ts, axis=0)
    
    if replace == 'mean':
        print("Replacing NaN points with feature mean value")
        
        # Getting Rid of NaN and Replacing with mean
        # Create list with average values of columns, excluding NaN values

        # Variable containing locations of NaN in data frame
        inds = np.where(np.isnan(data_process_tr)) 
        inds_t = np.where(np.isnan(data_process_ts)) 

        # Reassign locations of NaN to the column means
        data_process_tr[inds] = np.take(column_means, inds[1])
        data_process_ts[inds_t] = np.take(column_means_t, inds_t[1])
        
    if replace == 'median':
        print("Replacing NaN points with feature median value")
        
        # Getting Rid of NaN and Replacing with mean
        # Create list with average values of columns, excluding NaN values
        column_med = np.nanmedian(data_process_tr, axis=0)
        column_med_t = np.nanmedian(data_process_ts, axis=0)

        # Variable containing locations of NaN in data frame
        inds = np.where(np.isnan(data_process_tr)) 
        inds_t = np.where(np.isnan(data_process_ts)) 

        # Reassign locations of NaN to the column means
        data_process_tr[inds] = np.take(column_med, inds[1])
        data_process_ts[inds_t] = np.take(column_med_t, inds_t[1])
        
    if replace == 'zero':
        
        print("Replacing NaN points with zeros")
        # Variable containing locations of NaN in data frame
        inds = np.where(np.isnan(data_process_tr)) 
        inds_t = np.where(np.isnan(data_process_ts))
        
        data_process_tr[inds] = 0
        data_process_ts[inds_t] = 0
    
    print("Standerizing the data")
    column_std[column_std == 0] = 1
    data_process_tr = (data_process_tr - column_means)/column_std
    data_process_ts = (data_process_ts - column_means)/column_std

    return (data_process_tr, data_process_ts)

def transform_data(data, data_t, y, y_t, poly = 4, interact = True, log = True, three = True, pca_t = 1):

    # Build polynomial of degree poly
    print("Building polynomial of degree {0}".format(poly))
    data_tr = build_poly(data, poly)
    data_ts = build_poly(data_t, poly)
    
    if interact:
        # Build interaction terms
        
        print("Building the interactive terms")
        
        data_tr_int = build_interact_terms(data)
        data_ts_int = build_interact_terms(data_t)
        data_tr = np.c_[data_tr, data_tr_int]
        data_ts = np.c_[data_ts, data_ts_int]

    if log:
        # Build log 
        
        print("Taking the log value of the data")
        
        data_tr_log = np.log(abs(data)+1)
        data_ts_log = np.log(abs(data_t)+1)
        data_tr = np.c_[data_tr, data_tr_log]
        data_ts = np.c_[data_ts, data_ts_log]
        
    if three:
        # Build interaction terms
        
        print("Building the interactive terms or order three")
        
        data_tr_3 = build_three_way(data)
        data_ts_3 = build_three_way(data_t)
        data_tr = np.delete(data_tr, np.s_[0:30],axis=1)
        data_ts = np.delete(data_ts, np.s_[0:30],axis=1)
        data_tr = np.c_[data_tr, data_tr_3]
        data_ts = np.c_[data_ts, data_ts_3]
        

    # Perform PCA
    print("Performing PCA and keeping feature explaining {0} of the variance".format(pca_t))
    
    eigVal, eigVec, sumEigVal = PCA(data_tr, threshold = pca_t)
    data = data_tr.dot(eigVec)
    data_t = data_ts.dot(eigVec)
    print("Reducing the number of PCA to {0}".format(eigVec.shape[1]))
    
    
    print("Adding a columns of ones to the dataset")
    y, tx = build_model_data(data, y)
    y_t, tx_t = build_model_data(data_t, y_t)
    
    return y, tx, y_t, tx_t
