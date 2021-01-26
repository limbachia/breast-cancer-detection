import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Remove outliers
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, dictionary):
        self.dictionary = dictionary
        
    def fit(self,X, y = None):
        # only to accomodate the sklearn pipeline
        return self
    def transform(self,X):
        X = X.copy()
        for var, val in self.dictionary.items():
            X = X[X[var]<=val]
        return X

# Logarithm transform
class LogTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # This step is not needed, but to accomodate the pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = np.log1p(X[feature])
            
        return X


    
