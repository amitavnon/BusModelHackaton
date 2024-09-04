from __future__ import annotations
import numpy as np

from hackathon_code.models.base_model import BaseModel  # Assuming you have a BaseModel class
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline  # More elegant for preprocessing


class PolynomialRegression(BaseModel):
    def __init__(self, degree: int = 3, regularization=None, alpha=1.0, ignoreEstimated: bool = False):
        """
        Polynomial Regression Model

        Args:
            degree (int, optional): Degree of the polynomial. Defaults to 3.
            regularization (str, optional): Type of regularization ('l1', 'l2', or None). 
                                            Defaults to None.
            alpha (float, optional): Regularization strength. Defaults to 1.0.
        """

        super().__init__(f'Polynomial Regression-{degree}', ignoreEstimated)
        self.degree = degree

        # Create a pipeline for scaling and polynomial features
        self.pipeline = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=self.degree, include_bias=False)
        )

        # Choose the appropriate regression model with regularization
        if regularization == 'l1':
            from sklearn.linear_model import Lasso
            self.model = Lasso(alpha=alpha)
        elif regularization == 'l2':
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LinearRegression()

    def _fit(self, X, y) -> PolynomialRegression:
        """
        Fit the model to the data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target variable.

        Returns:
            PolynomialRegression: Returns self.
        """
        # Fit the pipeline (scaling and polynomial transformation)
        self.pipeline.fit(X)
        X_transformed = self.pipeline.transform(X)

        # Fit the regression model
        self.model.fit(X_transformed, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        # Transform the input features using the fitted pipeline
        X_transformed = self.pipeline.transform(X)
        return self.model.predict(X_transformed)