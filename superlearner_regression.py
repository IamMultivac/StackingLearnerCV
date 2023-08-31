from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import numpy as np
import pdb
import pandas as pd


class SuperlearnerRegression(BaseEstimator, RegressorMixin):

    def __init__(self, estimators, meta_estimator=None, use_initial_data=False, cv=5, random_state=1990):
        """
        TODO: write proper documentation

        :param estimators: list. Lists of regressors to be fitted on data
        :param meta_estimator: scikit-learn regressor. Model to be used as final estimator
        :param use_initial_data:
        :param cv: int. cross validation estrategy
        :param random_state: int. Number to be used as seed
        """
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.use_initial_data = use_initial_data
        self.random_state = random_state
        self.cv = cv



    def get_models_name(self):
        model_names = []
        for model in self.estimators:
            name = str(model)[:str(model).find('(')]
            model_names.append(name)

        self.models_name = model_names
        
        return self.models_name


    def fit(self, X, y, shuffle=True):
        self.X = X
        self.y = y

        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.to_numpy()
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.to_numpy()

        kfold = KFold(n_splits=self.cv, shuffle=shuffle, random_state=self.random_state)

        folds, self.meta_X, self.X_initial, self.meta_y = [], [], [], []

        for train_idx, test_idx in kfold.split(self.X, self.y):
            X_train, y_train, X_test, y_test = self.X[train_idx], self.y[train_idx], self.X[test_idx], self.y[test_idx]
            self.meta_y.append(y_test.reshape(-1, 1))
            self.X_initial.append(X_test)

            for model in self.estimators:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test).reshape(-1, 1)
                folds.append(predictions)
            self.meta_X.append(np.hstack(folds))
            folds = []
        self.meta_X = np.vstack(self.meta_X)
        self.meta_y = np.vstack(self.meta_y)

        self.X_initial = np.vstack(self.X_initial)

        if self.meta_estimator is None:
            self.meta_estimator = LinearRegression()

        if self.use_initial_data:
            self.meta_X = self._make_meta_X(self.X_initial, self.meta_X)

        self.meta_estimator.fit(self.meta_X, self.meta_y)

        for model in self.estimators:
            model.fit(self.X, self.y)
 
        return self
    
        
    def get_permutation_importances(self,scoring = None):
        p_importances = permutation_importance(estimator = self.meta_estimator, X = self.meta_X, y = self.meta_y, scoring = scoring, random_state= self.random_state)
        
        return p_importances


    def _make_meta_X(self, matrix_I, matrix_II):
        matrix = np.array([])
        if self.use_initial_data:
            matrix = np.hstack([matrix_I, matrix_II])
        return matrix



    def predict(self, X):
        predictions = []
        for model in self.estimators:
            predictions.append(model.predict(X).reshape(-1, 1))
        predictions = np.hstack(predictions)


        if self.use_initial_data:
            X = self._make_meta_X(X, predictions)
        else:
            X = predictions

        meta_predictions = self.meta_estimator.predict(X)

        return meta_predictions


