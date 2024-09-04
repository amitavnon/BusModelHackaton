from __future__ import annotations
import abc

import numpy as np
from sklearn.metrics import mean_squared_error as mse


class BaseModel:
    def __init__(self, name: str, ignoreEstimated: bool = False):
        self.name = name
        self.ignoreEstimated = ignoreEstimated

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseModel:
        if self.ignoreEstimated:
            try:
                indexes = X[~X.arrival_is_estimated]
                X, y = X[indexes], y[indexes]
            except:
                pass
        return self._fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self._predict(X)
        if self.ignoreEstimated:
            try:
                predictions[~X.arrival_is_estimated] = 0
            except:
                pass

        return predictions

    @abc.abstractmethod
    def _fit(self, X, y) -> BaseModel:
        """
        Fit the model
        :param X: samples
        :param y: labels
        :return: self
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels from samples X
        :param X: samples
        :return: predictions
        """
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Returns the model's MSE score
        :param X:
        :param y:
        :return:
        """
        return mse(y, self.predict(X))
