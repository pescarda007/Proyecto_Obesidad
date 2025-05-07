from sklearn.base import BaseEstimator, TransformerMixin


class BMICalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["IMC"] = X["Weight"] / (X["Height"] ** 2)
        return X