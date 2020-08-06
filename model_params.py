from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lars
from sklearn.ensemble import ExtraTreesRegressor

SVR_MODEL = (SVR, {'kernel':['linear','poly','rbf','sigmoid'],
'degree':[3,4,5],
'C':[0.9, 0.95, 1.0, 1.05, 1.10],
'gamma':[0.001, 0.01, 0.1, 1],
'tol':[1e-3, 1e-4, 2e-3, 5e-3]
})

MLP_REGRESSOR = (MLPRegressor, {'hidden_layer_sizes': [(50,50,50), (100,)],
'activation': ['relu'],
'solver': ['adam'],
'alpha': [0.0001],
'learning_rate': ['constant'],
'max_iter': [300, 350],
'momentum': [0.9, 0.95]
})

RANDOM_FOREST_REGRESSOR = (RandomForestRegressor, {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]})


GRADIENT_BOOSTING_REGRESSOR = (GradientBoostingRegressor, {'loss': ['ls', 'lad', 'huber'],
'learning_rate': [0.05, 0.1, 0.15, 0.2],
'n_estimators': [100, 200, 300, 400, 500],
'min_samples_split': [2, 3, 4],
'min_samples_leaf': [1, 2, 3, 4],
'max_depth': [5, 10, 15, 20, 25, 30, 40],
'max_features': [5, 10, 20, 30, 40, 50, 60, 70]
})

LASSO_LARS = (LassoLars, {'alpha': [0.9, 0.95, 1.0, 1.05],
'fit_intercept': [True, False]})

EXTRA_TREE_REGRESSOR = (ExtraTreeRegressor, {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]})

KERNEL_RIDGE = (KernelRidge, {'alpha': [0.9, 0.95, 1.0, 1.05],
'kernel': ['chi2', 'linear', 'polynomial', 'rbf', 'laplacian'],
'degree': [2,3,4]})

RIDGE = (Ridge, {'alpha': [0.9, 0.95, 1.0, 1.05],
'solver': ['auto', 'svd', 'cholesky', 'lsqr']})

LASSO = (Lasso, {'alpha': [0.9, 0.95, 1.0, 1.05],
'fit_intercept': [True, False]})

MULTITASK_LASSO = (MultiTaskLasso, {'alpha': [0.9, 0.95, 1.0, 1.05],
'fit_intercept': [True, False]})

ELASTIC_NET = (ElasticNet, {'alpha': [0.9, 0.95, 1.0, 1.05],
'fit_intercept': [True, False]})

LINEAR_REGRESSION = (LinearRegression, {'fit_intercept': [True, False]})

LARS = (Lars, {'fit_intercept': [True, False]})


MODEL_DICT = {"Linear Regression": LINEAR_REGRESSION,
            #   "SVR": SVR_MODEL, 
            #   "MLP Regressor": MLP_REGRESSOR,
            #   "Random Forest Regressor": RANDOM_FOREST_REGRESSOR,
              "Gradient Boosting Regressor": GRADIENT_BOOSTING_REGRESSOR,
            #   "Lasso Lars": LASSO_LARS,
            #   "Extra Tree Regressor": EXTRA_TREE_REGRESSOR,
            #   "Kernel Ridge": KERNEL_RIDGE,
            #   "Ridge": RIDGE,
            #   "Lasso": LASSO,
            #   "Multitask Lasso": MULTITASK_LASSO,
            #   "Elastic Net": ELASTIC_NET,
            #   "Lars": LARS}
            }