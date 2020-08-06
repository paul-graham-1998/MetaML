# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:25:57 2019

@author: PaulGraham
"""
import sklearn
import sys
import pandas as pd
#import tensorflow as tf
import ast
import re
import numpy as np
from sklearn.model_selection import KFold
from data_architecture import Architecture
def zero_pad(data_list):
    padded_data_list= np.zeros([len(data_list),24])
    for i,j in enumerate(data_list):
        padded_data_list[i][0:len(j)] = j
    return padded_data_list

class data_struct:
    def __init__(self , path, has_labels):
        self.df = pd.read_csv(path)
        self.number_of_rows = self.df.shape[0]
        self.id = self.id = np.reshape(np.array(self.df['id']),(self.number_of_rows,1))

        #`arch_and_hp` is architecture and hyperparameters
        self.arch_and_hp = [0]*self.number_of_rows
        count = 0
        for arch_and_hp in self.df['arch_and_hp']:
            self.arch_and_hp[count] = Architecture(arch_and_hp)
            count += 1

        #Collection of features that are floating points and can be treated simply
        self.batch_size_test = np.reshape(np.array(self.df['batch_size_test']),(self.number_of_rows,1))
        self.batch_size_val = np.reshape(np.array(self.df['batch_size_val']),(self.number_of_rows,1))
        self.criterion = np.reshape(np.array(self.df['criterion']),(self.number_of_rows,1))
        self.epochs = np.reshape(np.array(self.df['epochs']),(self.number_of_rows,1))
        self.number_parameters = np.reshape(np.array(self.df['number_parameters']),(self.number_of_rows,1))
        self.optimizer = np.reshape(np.array(self.df['optimizer']),(self.number_of_rows,1))        
        self.batch_size_train =  np.reshape(np.array(self.df['batch_size_train']),(self.number_of_rows,1))

        #Average of the initial parameters. Zero padding since the number of initial parameters can vary.
        self.init_params_mu = []
        for init_params_mu in self.df['init_params_mu']:
            try:
                slice_data = init_params_mu[1:-2]
                strip_data = slice_data.split(', ')
                float_data = [float(string) for string in strip_data]
                self.init_params_mu.append(float_data)
            except:
                self.init_params_mu.append([]) 
        self.init_params_mu = zero_pad(self.init_params_mu)

        #Standard deviation of the initial parameters. Zero padding since the number of initial parameters can vary.
        self.init_params_std = []
        for init_params_std in self.df['init_params_std']:
            try:
                slice_data = init_params_std[1:-2]
                strip_data = slice_data.split(', ')
                float_data = [float(string) for string in strip_data]
                self.init_params_std.append(float_data)
            except:
                self.init_params_std.append([])      
        self.init_params_std = zero_pad(self.init_params_std)

        #L2 norm of the initial parameters. Zero padding since the number of initial parameters can vary.
        self.init_params_l2 = []
        for init_params_l2 in self.df['init_params_l2']:
            try:
                slice_data = init_params_l2[1:-2]
                strip_data = slice_data.split(', ')
                float_data = [float(string) for string in strip_data]
                self.init_params_l2.append(float_data)
            except:
                self.init_params_l2.append([])
        self.init_params_l2 = zero_pad(self.init_params_l2)

        #Validation error for the 50 first epochs
        self.val_errors = []
        for idx in range(0,50):
            label_val_errors = 'val_accs_' + str(idx)
            for model in self.df[label_val_errors]:
                self.val_errors.append(float(model))
        self.val_errors = np.reshape(np.array(self.val_errors) , (self.number_of_rows,50))

        #Validation losses for the 50 first epochs
        self.val_losses = []
        for idx in range(0,50):
            label_val_losses = 'val_losses_' + str(idx)
            for model in self.df[label_val_losses]:
                self.val_losses.append(float(model))
        self.val_losses = np.reshape(np.array(self.val_losses) , (self.number_of_rows,50))

        #Training error for the 50 first epochs
        self.train_errors = []
        for idx in range(0,50):
            label_train_errors = 'train_accs_' + str(idx)
            for model in self.df[label_train_errors]:
                self.train_errors.append(float(model))
        self.train_errors = np.reshape(np.array(self.train_errors) , (self.number_of_rows,50))

        #Training losses for the 50 first epochs
        self.train_losses = []
        for idx in range(0,50):
            label_train_losses = 'train_losses_' + str(idx)
            for model in self.df[label_train_losses]:
                self.train_losses.append(float(model))
        self.train_losses = np.reshape(np.array(self.train_losses) , (self.number_of_rows,50))

        #Average values of the 50 epochs of training and validation data.
        self.val_errors_avg = np.reshape(np.mean(self.val_errors, axis = 1),(self.number_of_rows,1))
        self.val_losses_avg = np.reshape(np.mean(self.val_losses, axis = 1),(self.number_of_rows,1))
        self.train_errors_avg = np.reshape(np.mean(self.train_errors, axis = 1),(self.number_of_rows,1))
        self.train_losses_avg = np.reshape(np.mean(self.train_losses, axis = 1),(self.number_of_rows,1))

        #Last values of the 50 epochs of training and validation data.
        self.val_errors_end = np.reshape(self.val_errors[:,-1],(self.number_of_rows,1))
        self.val_losses_end = np.reshape(self.val_losses[:,-1],(self.number_of_rows,1))
        self.train_errors_end = np.reshape(self.train_errors[:,-1],(self.number_of_rows,1))
        self.train_losses_end = np.reshape(self.train_losses[:,-1],(self.number_of_rows,1))

        #Will be used as training data features 
        if has_labels:
            self.val_error =  np.reshape(np.array(self.df['val_error']),(self.number_of_rows,1))
            self.val_loss =  np.reshape(np.array(self.df['val_loss']),(self.number_of_rows,1))
            self.train_error =  np.reshape(np.array(self.df['train_error']),(self.number_of_rows,1))
            self.train_loss =  np.reshape(np.array(self.df['train_loss']),(self.number_of_rows,1))

    def get_id(self):
        return self.id
    def get_arch_and_hp(self):
        return self.arch_and_hp
    def get_batch_size_test(self):
        return self.batch_size_test
    def get_batch_size_val(self):
        return self.batch_size_val
    def get_criterion(self):
        return self.criterion
    def get_epochs(self):
        return self.epochs
    def get_number_parameters(self):
        return self.number_parameters
    def get_optimizer(self):
        return self.optimizer
    def get_val_error(self):
        return self.val_error
    def get_val_loss(self):
        return self.val_loss
    def get_train_error(self):
        return self.train_error
    def get_train_loss(self):
        return self.train_loss
    def get_batch_size_train(self):
        return self.batch_size_train
    def get_init_params_mu(self):
        return self.init_params_mu
    def get_init_params_std(self):
        return self.init_params_std
    def get_init_params_l2(self):
        return self.init_params_l2
    def get_val_errors(self):
        return self.val_errors
    def get_val_losses(self):
        return self.val_losses
    def get_train_errors(self):
        return self.train_errors
    def get_train_losses(self):
        return self.train_losses

    def get_val_errors_avg(self):
        return self.val_errors_avg
    def get_val_losses_avg(self):
        return self.val_losses_avg
    def get_train_errors_avg(self):
        return self.train_errors_avg
    def get_train_losses_avg(self):
        return self.train_losses_avg

    def get_val_errors_end(self):
        return self.val_errors_end
    def get_val_losses_end(self):
        return self.val_losses_end
    def get_train_errors_end(self):
        return self.train_errors_end
    def get_train_losses_end(self):
        return self.train_losses_end


    def get_number_layers(self):
        return np.reshape(np.array([len(network.layer_list) for network in self.arch_and_hp]) ,(self.number_of_rows,1))

    def get_number_tanh(self):
        return np.reshape(np.array([network.number_tanh for network in self.arch_and_hp]) ,(self.number_of_rows,1))
    def get_number_flatten(self):
        return np.reshape(np.array([network.number_flatten for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_linear(self):
        return np.reshape(np.array([network.number_linear for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_batchnorm1D(self):
        return np.reshape(np.array([network.number_batchnorm1D for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_selu(self):
        return np.reshape(np.array([network.number_selu for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_softmax(self):
        return np.reshape(np.array([network.number_softmax for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_batchnorm(self):
        return np.reshape(np.array([network.number_batchnorm for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_dropout(self):
        return np.reshape(np.array([network.number_dropout for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_leaky_relu(self):
        return np.reshape(np.array([network.number_leaky_relu for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_maxpool(self):
        return np.reshape(np.array([network.number_maxpool for network in self.arch_and_hp]),(self.number_of_rows,1))
    def get_number_relu(self):
        return np.reshape(np.array([network.number_relu for network in self.arch_and_hp]),(self.number_of_rows,1))

    def set_feature_list(self, feature_inclusion_dict):
        self.inclusion_list = [
            (self.get_number_layers(), feature_inclusion_dict["include_num_of_layers"]),
            (self.get_number_tanh(), feature_inclusion_dict["include_tanh"]),
            (self.get_number_flatten(), feature_inclusion_dict["include_flatten"]),
            (self.get_number_linear(), feature_inclusion_dict["include_linear"]),
            (self.get_number_batchnorm1D(), feature_inclusion_dict["include_batchnorm1d"]),
            (self.get_number_selu(), feature_inclusion_dict["include_selu"]),
            (self.get_number_softmax(), feature_inclusion_dict["include_batchnorm"]),
            (self.get_number_dropout(), feature_inclusion_dict["include_dropout"]),
            (self.get_number_relu(), feature_inclusion_dict["include_relu"]),
            (self.get_number_leaky_relu(), feature_inclusion_dict["include_leaky_relu"]),
            (self.get_number_maxpool(), feature_inclusion_dict["include_maxpool"]),
            (self.get_train_errors_avg(), feature_inclusion_dict["include_train_errors_avg"]),
            (self.get_val_errors_avg(), feature_inclusion_dict["include_validation_error_avg"]),
            (self.get_train_losses_avg(), feature_inclusion_dict["include_train_losses_avg"]),
            (self.get_val_losses_avg(), feature_inclusion_dict["include_validation_losses_avg"]),
            (self.get_train_errors_end(), feature_inclusion_dict["include_train_errors_end"]),
            (self.get_val_errors_end(), feature_inclusion_dict["include_validation_errors_end"]),
            (self.get_train_losses_end(), feature_inclusion_dict["include_train_losses_end"]),
            (self.get_val_losses_end(), feature_inclusion_dict["include_validation_losses_end"]),
            (self.get_init_params_mu(), feature_inclusion_dict["include_params_mu"]),
            (self.get_init_params_std(), feature_inclusion_dict["include_params_std"]),
            (self.get_init_params_l2(), feature_inclusion_dict["include_params_l2"]),
            (self.get_val_errors(), feature_inclusion_dict["include_validation_errors"]),
            (self.get_val_losses(), feature_inclusion_dict["include_validation_losses"]),
            (self.get_train_losses(), feature_inclusion_dict["include_train_losses"]),
            (self.get_train_errors(), feature_inclusion_dict["include_train_errors"]),
            (self.get_epochs(), feature_inclusion_dict["include_epochs"])
        ]   

    def stack_features(self):
        to_include = tuple([el[0] for el in self.inclusion_list if el[1]])
        features = np.hstack(to_include)
        return features