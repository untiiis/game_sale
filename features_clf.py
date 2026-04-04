
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AddTopFeaturesClf(BaseEstimator, TransformerMixin):
    def __init__(self, y_global, top_n=20):
        self.top_n = top_n
        self.y_global = y_global
    
    def fit(self, X, y=None):
        X = X.copy()
        X['Global_Sales'] = self.y_global
        
        self.top_publishers_ = X.groupby('Publisher', observed=False)['Global_Sales'].sum().nlargest(self.top_n).index.tolist()
        self.top_platforms_ = X.groupby('Platform', observed=False)['Global_Sales'].sum().nlargest(self.top_n).index.tolist()
        self.top_genres_ = X.groupby('Genre', observed=False)['Global_Sales'].sum().nlargest(self.top_n).index.tolist()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Top_Publisher'] = X['Publisher'].apply(lambda x: x if x in self.top_publishers_ else 'Others')
        X['Top_Platform'] = X['Platform'].apply(lambda x: x if x in self.top_platforms_ else 'Others')
        X['Top_Genre'] = X['Genre'].apply(lambda x: x if x in self.top_genres_ else 'Others')
        X = X.drop(['Publisher', 'Platform', 'Genre'], axis=1)
        return X
