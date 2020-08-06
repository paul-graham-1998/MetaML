import sklearn
import sys
import pandas as pd
#import tensorflow as tf
import ast
import re
import numpy as np
from collections import Counter

class Architecture:
    def __init__(self,input_string):
        self.number_tanh = 0#Tanh
        self.number_flatten = 0#Flatten
        self.number_linear = 0#Linear
        self.number_batchnorm1D = 0#BatchNorm1D
        self.number_selu = 0#SELU
        self.number_softmax = 0#Softmax
        self.number_batchnorm = 0#batchnorm2D
        self.number_dropout = 0#dropout2D and Dropout
        self.number_leaky_relu = 0#LeakyRelu
        self.number_maxpool = 0#MaxPool2d
        self.number_relu = 0#ReLU


        LAYER_PATTERN = r"[A-za-z0-9]+[a-zA-Z](?=[0-9]+\)\:)"
        PARAMS_PATTERN = r"[A-za-z0-9]+\)\:\s"

        layer_compile = re.compile(LAYER_PATTERN)
        params_compile = re.compile(PARAMS_PATTERN)
        arch_and_hp = input_string[13:]
        layer_tokens = layer_compile.findall(arch_and_hp)
        params_tokens = [params[:-2] for params in params_compile.split(arch_and_hp)]
        self.layer_list = []
        for index in range(len(layer_tokens)):
            self.layer_list.append( (layer_tokens[index] , params_tokens[index+1]) )

        layer_type_list = [layer[0] for layer in self.layer_list]
        for layer in layer_type_list:
            if layer == 'tanh':
                self.number_tanh += 1
            elif layer == 'flatten':
                self.number_flatten += 1
            elif layer == 'linear':
                self.number_linear += 1
            elif layer == 'batchnorm1D':
                self.number_batchnorm1D += 1
            elif layer == 'selu':
                self.number_selu += 1
            elif layer == 'softmax':
                self.number_softmax += 1
            elif layer == 'batchnorm':
                self.number_batchnorm += 1
            elif layer == 'dropout':
                self.number_dropout += 1
            elif layer == 'leaky_relu':
                self.number_leaky_relu += 1
            elif layer == 'maxpool':
                self.number_maxpool += 1
            elif layer == 'relu':
                self.number_relu += 1
