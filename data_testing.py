"""
Created on Thu Nov  7 03:09:56 2019

@author: PaulGraham
"""

import sklearn
import pandas as pd


from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import model_params as mp




def random_search(model, params_dist, features, labels, number_of_iterations):
    random_search_model = RandomizedSearchCV(estimator=model(),
    param_distributions=params_dist,
    scoring='r2',
    n_iter=number_of_iterations,
    cv=2,
    n_jobs=-1)

    random_search_model.fit(features, labels)
    params = random_search_model.best_params_ 
    score = random_search_model.score(features, labels)
    print(score)
    print(params)
    
def grid_search(model, params_grid, features, labels):
    grid_search_model = GridSearchCV(estimator=model(),
    param_grid=params_grid,
    scoring='r2',
    cv=2,
    n_jobs=-1)

    grid_search_model.fit(features, labels)
    params = grid_search_model.best_params_ 
    grid_search_model.set_params(**params)
    score = grid_search_model.score(features, labels)
    print(score)
    return grid_search_model

def multi_grid_search(features, labels):
    for model_name in mp.MODEL_DICT:
        print(model_name)
        model, params_dict = mp.MODEL_DICT[model_name]
        grid_search(model, params_dict, features, labels, number_of_iterations)


def multi_random_search(features, labels, number_of_iterations):
    for model_name in mp.MODEL_DICT:
        print(model_name)
        AP = mp.MODEL_DICT[model_name]
        random_search(model, params_dict, features, labels, number_of_iterations)
          
def get_best_features(model_obj, features, labels):
    selector = RFECV(model_obj, step=1, cv=5)
    selector = selector.fit(features, labels)
    print(selector.support_)
    print(selector.ranking_)
     
def set_model_params(model, params):
    model = model(**params)    
    return model

def fit_model(model_obj, features, labels):
    model.fit(features, labels)
    return model

