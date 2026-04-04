
import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

# Classe features simple
class AddBasicFeaturesClass(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self                   # pas de calcul nécessaire au fit
    def transform(self, X):
        X = X.copy()
        X['Decade'] = (X['Year_of_Release'] // 10) * 10
        X['Decade'] = X['Decade'].astype('category')
        X['Has_User_Score'] = X['User_Score'].apply(lambda x: 0 if pd.isna(x) else 1)
        X['Has_Critic_Score'] = X['Critic_Score'].apply(lambda x: 0 if pd.isna(x) else 1)
        X['Game_Age'] = 2025 - X['Year_of_Release']
        X['Score_Avg'] = (X['User_Score'] + X['Critic_Score']) / 2
        X['Score_Product'] = X['User_Score'] * X['Critic_Score']
        return X

# Classe features qui dependent des autres
class AddTopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=20):
        self.top_n = top_n
    
    def fit(self, X, y=None):
        X = X.copy()
        X['Global_Sales'] = y 
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

# Imputation hiérarchique
class HierarchicalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['Critic_Score', 'User_Score']):
        self.cols = cols

    def fit(self, X, y=None):
        self.global_median_ = {col: X[col].median() for col in self.cols}
        self.platform_median_ = {col: X.groupby('Top_Platform', observed=False)[col].median() for col in self.cols}
        self.genre_median_ = {col: X.groupby('Top_Genre', observed=False)[col].median() for col in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(X['Top_Platform'].map(self.platform_median_[col]))
            X[col] = X[col].fillna(X['Top_Genre'].map(self.genre_median_[col]))
            X[col] = X[col].fillna(self.global_median_[col])
        return X
