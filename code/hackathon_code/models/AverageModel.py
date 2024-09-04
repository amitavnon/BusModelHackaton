import numpy as np

from hackathon_code.models.base_model import BaseModel


class AverageModel(BaseModel):
    def __init__(self, ignore_estimated: bool = False):
        super().__init__('Average', ignore_estimated)
        self.average = None

    def _fit(self, X, y):
        self.average = np.average(y)
        return self

    def _predict(self, X):
        return np.ones(shape=(len(X))) * self.average
