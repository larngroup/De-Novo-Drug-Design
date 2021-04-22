# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def grid_search(data,model_type):
    """
    This function performs a grid-search to find the best parameters of each 
    standard QSAR model.
    ----------
    data: Array with training data and respectives labels
    model_type: String that indentifies the model (SVR, RF, and KNN)
    Returns
    -------
    This function returns the best parameters of the experimented model, to be 
    then used in the implementation of the optimal model
    """
    
    X_train = data[0]
    y_train = data[1]
    
    if model_type == 'SVR':
       
        model = SVR()
        gs = GridSearchCV(model, {'kernel': ['rbf','linear','poly'], 'C': 2.0 ** np.array([-3, 13]), 'gamma': 2.0 ** np.array([-13, 3])}, n_jobs=5)
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
        
    elif model_type == 'RF':
        
        model = RandomForestRegressor()
        gs = GridSearchCV(model, {'n_estimators': [500,1000,1500],'max_features': ["auto", "sqrt", "log2"]})
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
  
    elif model_type == 'KNN':
        
        
        model = KNeighborsRegressor()
        gs = GridSearchCV(model, {'n_neighbors': [3,5,9,11], 'metric': ['euclidean', 'manhattan', 'chebyshev']})
        gs.fit(X_train, y_train)
        params = gs.best_params_
        print(params)
        
       
    return params