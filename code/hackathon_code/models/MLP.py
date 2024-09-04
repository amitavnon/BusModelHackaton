from __future__ import annotations

from sklearn.preprocessing import StandardScaler

from hackathon_code.models.base_model import BaseModel
from sklearn.neural_network import MLPRegressor


class MLP(BaseModel):
    def __init__(self, ignore_estimated: bool = False):
        super().__init__('MLP', ignore_estimated)
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50),  # You can adjust the architecture
                                  activation='relu',  # Activation function
                                  solver='adam',  # Optimization algorithm
                                  alpha=0.001,  # Regularization term (L2 penalty)
                                  batch_size='auto',  # Size of minibatches
                                  learning_rate='adaptive',  # Keeps the learning rate constant to adaptive
                                  learning_rate_init=0.005,  # Initial learning rate
                                  max_iter=500,  # Maximum number of iterations
                                  shuffle=True,  # Shuffle samples in each iteration
                                  random_state=42,  # Seed for reproducibility
                                  tol=1e-4,  # Tolerance for optimization
                                  verbose=True)
        self.scaler = StandardScaler()

    def _fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def _predict(self, X):
        return self.model.predict(self.scaler.fit_transform(X))