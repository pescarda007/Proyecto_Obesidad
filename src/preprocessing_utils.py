from sklearn.base import BaseEstimator, TransformerMixin


class BMICalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["IMC"] = X["Weight"] / (X["Height"] ** 2)
        return X
    
class CustomReplacer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        bool_map = {'yes': 1, 'no': 0}
        caec_calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        for col in ['family_history_with_overweight', 'FAVC', 'SMOKE']:
            X[col] = X[col].replace(bool_map).astype(int)

        X['CALC'] = X['CALC'].replace(caec_calc_map).astype(int)

        #if 'NObeyesdad' in X.columns:
        #X['NObeyesdad'] = X['NObeyesdad'].replace(target_map).astype(int)

        return X
