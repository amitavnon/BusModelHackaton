from hackathon_code.models.base_model import BaseModel
from sklearn.linear_model import LinearRegression as LR


class LinearRegression(BaseModel):
    def __init__(self, ignore_estimated: bool = False):
        super().__init__('Linear Regression', ignore_estimated)
        self.model = LR()

    def _fit(self, X, y):
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        return self.model.predict(X)