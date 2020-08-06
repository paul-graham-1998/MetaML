import sklearn
import sys
import pandas as pd
import tensorflow as tf
import csv
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def write_data(testing_ids , testing_labels_1 , testing_labels_2):
    with open("C:\\Users\\paulg\\OneDrive\\Documents\\pythonScripts\\446ML\\results.csv", mode = 'w') as labels_file:
        label_writer = csv.writer(labels_file, delimiter = ',', lineterminator = '\n')
        label_writer.writerow(['Id', 'Predicted'])
        for index in range(len(testing_ids)):
            id_1 = 'test_' + testing_ids[index][5:] + '_val_error'
            id_2 = 'test_' + testing_ids[index][5:] + '_train_error'
            label_writer.writerow([ id_1 , testing_labels_2[index] ])
            label_writer.writerow([ id_2 , testing_labels_1[index] ])
    labels_file.close()


