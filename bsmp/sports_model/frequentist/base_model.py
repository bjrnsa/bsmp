from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for predictive models.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """
        Fit the model to the data.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes based on the fitted model.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict match outcome probabilities.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Get the current parameters of the model.
        """
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """
        Set parameters for the model.
        """
        pass

    @abstractmethod
    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.
        """
        pass

    def _validate_X(self, X: pd.DataFrame, fit: bool = True) -> None:
        """
        Validate input DataFrame dimensions and types.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        fit : bool, default=True
            Whether this is being called during fit (requires at least 2 columns)
            or during predict (requires exactly 2 columns)
        """
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Check minimum number of columns
        min_cols = 2
        if X.shape[1] < min_cols:
            raise ValueError(f"X must have at least {min_cols} columns")

        # For predict methods, exactly 2 columns are required
        if not fit and X.shape[1] != 2:
            raise ValueError("X must have exactly 2 columns for prediction")

        # Check that first two columns contain strings (team names)
        for i in range(2):
            if not pd.api.types.is_string_dtype(X.iloc[:, i]):
                raise ValueError(f"Column {i} must contain string values (team names)")

    def _validate_teams(self, teams: List[str]) -> None:
        """Validate teams exist in the model."""
        for team in teams:
            if team not in self.team_map_:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
