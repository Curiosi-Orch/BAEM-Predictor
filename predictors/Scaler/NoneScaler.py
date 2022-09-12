from sklearn.base import TransformerMixin, BaseEstimator

class NoneScaler(TransformerMixin, BaseEstimator):
    """ A scaler with no process on the data
    """ 
    def __init__(self):
      pass

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X