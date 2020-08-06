"""
Created on Thu Nov  7 03:09:56 2019

@author: PaulGraham
"""

from model_params import GRADIENT_BOOSTING_REGRESSOR
from data_builder import build_training_data
from data_builder import build_testing_data
import data_testing as dt
from write_results import write_data
from data_clean import l2_normalize


(train_validation_features, train_validation_labels) , (train_training_features, train_training_labels) = build_training_data()
test_ids, test_validation_features, test_training_features = build_testing_data()
train_training_features, test_training_features = l2_normalize(train_training_features, test_training_features)
train_validation_features, test_validation_features = l2_normalize(train_validation_features, test_validation_features)

# dt.multi_random_search(train_training_features, train_training_labels)
# model, param_grid = GRADIENT_BOOSTING_REGRESSOR
# trained_model = dt.random_search(model, param_grid, train_training_features, train_training_labels,20)


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor
params = {'n_estimators': 400, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 50, 'max_depth': 15, 'loss': 'lad', 'learning_rate': 0.15}
model_obj = dt.set_model_params(model, params)
dt.get_best_features(model_obj, train_training_features, train_training_labels)



