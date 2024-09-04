from __future__ import annotations

from hackathon_code.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor


class RandomForest(BaseModel):
    def __init__(self, depth=None, ignore_estimated=False):
        super().__init__(f'RandomForest-{depth if depth is not None else "*"}', ignore_estimated)
        self.model = RandomForestRegressor(max_depth=depth, random_state=69)

    def _fit(self, X, y):
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        return self.model.predict(X)