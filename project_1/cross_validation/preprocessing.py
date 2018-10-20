import numpy as np

def process_data(data, data_t, labels, ids ,sample_filtering = True, feature_filtering = True, remove_outlier = True, replace = 'zero'):
    """The process_data function prepares the train and test data by performing
    some data cleansing techniques. Missing values which are set as -999
    are replaced by NaN, then the means of each features are calculated
    in order to replace NaN values by the means. Features and samples with
    too much missing values are deleted by this function. Returns filtered
    dataset and column idx to be removed as well as column means and variance
    (required for transformation of unseen data."""

    
    data_process_tr = np.array(data)
    data_process_ts = np.array(data_t)
    lab = np.array(labels)
    
    # Setting missing values (-999) as NaN
    data_process_tr[data_process_tr == -999] = np.nan
    data_process_ts[data_process_ts == -999] = np.nan
    
    print('The original dimensions of the training data set was {0} samples and {1} columns'.format(data_process_tr.shape[0],data_process_tr.shape[1]))
    

    # Filtering weak features and samples

    # Retrieving percentage for each feature 
    
    if feature_filtering:
        nan_count = np.count_nonzero(np.isnan(data_process_tr),axis=0)/data_process_tr.shape[0]
        
        idx_del = []
        
        for idx, entry in enumerate(nan_count):
            if (entry >= 0.6):
                idx_del.append(idx)
        data_process_tr = np.delete(data_process_tr, idx_del,1).copy()
        data_process_ts = np.delete(data_process_ts, idx_del,1).copy()
        
    if sample_filtering:
        data_set_filtered_2 = [] # Dataset filtered for features and samples
        
        # Arrays that gets rid of entries that are no longer corresponding in the dataframe
        sample_del = []
        
        # Retrieve percentage for each sample
        nan_count = np.count_nonzero(np.isnan(data_process_tr),axis=1)/data_process_tr.shape[1]
        
        # Gets rid of samples with NaN values higher than 15%
        for idx, entry in enumerate(nan_count):  
            if entry >= 0.15:
                sample_del.append(idx)
                
        data_process_tr = np.delete(data_process_tr,sample_del,0).copy()
        lab = np.delete(lab,sample_del,0).copy()


    # Print new dimensions of the dataframe after filtering

    print(' After feature and sample filtering, there are {0} samples and {1} columns'.format(data_process_tr.shape[0],data_process_tr.shape[1]))
    
    if replace == 'mean':
        # Getting Rid of NaN and Replacing with mean
        # Create list with average values of columns, excluding NaN values
        column_means = np.nanmean(data_process_tr, axis=0)
        column_means_t = np.nanmean(data_process_ts, axis=0)

        # Variable containing locations of NaN in data frame
        inds = np.where(np.isnan(data_process_tr)) 
        inds_t = np.where(np.isnan(data_process_ts)) 

        # Reassign locations of NaN to the column means
        data_process_tr[inds] = np.take(column_means, inds[1])
        data_process_ts[inds_t] = np.take(column_means_t, inds_t[1])
        
    if replace == 'zero':
        

        # Variable containing locations of NaN in data frame
        inds = np.where(np.isnan(data_process_tr)) 
        inds_t = np.where(np.isnan(data_process_ts))
        
        data_process_tr[inds] = 0
        data_process_ts[inds_t] = 0
        
    if remove_outlier:

        column_means = np.mean(data_process_tr, axis=0)
        column_means_t = np.mean(data_process_ts, axis=0)

        column_std = np.std(data_process_tr, axis=0)
        column_std_t = np.std(data_process_ts, axis=0)
        
        data_tr = np.zeros(data_process_tr.shape)
        data_ts = np.zeros(data_process_ts.shape)
        
        for i in range(len(column_means)):
            col = data_process_tr[:,i]
            thresh_sup = column_means[i] + 3*column_std[i]
            thresh_inf = column_means[i] - 3*column_std[i]
            
            col_t = data_process_ts[:,i]
            thresh_ts = column_means_t[i] + 3*column_std_t[i]
            thresh_ti = column_means_t[i] - 3*column_std_t[i]
            
            col[col >= thresh_sup] = column_means[i]
            col[col <= thresh_inf] = column_means[i]
            
            col_t[col_t >= thresh_ts] = column_means[i]
            col_t[col_t <= thresh_ti] = column_means[i]

            data_tr[:,i] = col
            
            data_ts[:,i] = col_t
        
        data_process_tr = data_tr.copy()
        data_process_ts = data_ts.copy()
    return (data_process_tr, data_process_ts, lab)
