import sklearn
import sys
import pandas as pd
from pathlib import Path
#import tensorflow as tf
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from feature_selection import VALIDATION_ERROR_FEATURES, TRAINING_ERROR_FEATURES
from data_struct import data_struct

TRAIN_PATH = Path(__file__).resolve().parent / "data/train.csv"
TEST_PATH = Path(__file__).resolve().parent / "data/test.csv"

#Read the training and testing data
def build_training_data():
    train_struct = data_struct(path=TRAIN_PATH, has_labels=True)

    train_struct.set_feature_list(VALIDATION_ERROR_FEATURES)
    train_validation_features = train_struct.stack_features()
    train_validation_labels = train_struct.get_val_error()

    train_struct.set_feature_list(TRAINING_ERROR_FEATURES)
    train_training_features = train_struct.stack_features()
    train_training_labels = train_struct.get_train_error()
    
    sample_size, num_features = train_validation_features.shape
    train_training_labels = np.reshape(train_training_labels,(sample_size,))
    train_validation_labels = np.reshape(train_validation_labels,(sample_size,))
    
    return [(train_validation_features, train_validation_labels) , (train_training_features, train_training_labels)]

def build_testing_data():
    test_struct = data_struct(path=TEST_PATH, has_labels=False)
    test_ids = test_struct.get_id()
    test_struct.set_feature_list(VALIDATION_ERROR_FEATURES)
    test_validation_features = test_struct.stack_features()

    test_struct.set_feature_list(TRAINING_ERROR_FEATURES)
    test_training_features = test_struct.stack_features()
    return test_ids, test_validation_features, test_training_features

