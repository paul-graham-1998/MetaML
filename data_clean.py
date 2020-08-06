import numpy as np
from sklearn.preprocessing import normalize

def l2_normalize(training_data , testing_data):
    split = training_data.shape[0]
    all_data = np.vstack((training_data , testing_data))
    norm_data = normalize(all_data, norm='l2', axis=0, copy=True, return_norm=False)
    norm_training_data = norm_data[:split,:]
    norm_testing_data = norm_data[split:,:]
    return norm_training_data , norm_testing_data

def split(data, training_frac):
    cutting_index = int(training_frac * data.shape[0])
    training_data = data[:cutting_index,:]
    validation_data = data[cutting_index:,:]
    return training_data , validation_data

def add_to(data, new_data):
     return np.hstack(data, new_data)
