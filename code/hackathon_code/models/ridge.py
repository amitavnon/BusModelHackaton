
from hackathon_code.models.base_model import BaseModel
from sklearn.linear_model import Ridge


class RidgeModel(BaseModel):
    def __init__(self, ignore_estimated: bool = False):
        super().__init__('Ridge', ignore_estimated)
        self.model = Ridge(alpha=1.0)

    def _fit(self, X, y):
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        return self.model.predict(X)