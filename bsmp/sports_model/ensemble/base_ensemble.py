"""Base class for ensemble models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bsmp.sports_model.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry, Poisson


class BaseEnsemble(ABC):
    """Abstract base class for ensemble models that combine predictions from multiple sports models.

    Attributes:
        models (Dict[str, Union[BradleyTerry, GSSD, PRP, TOOR, ZSD]]): Dictionary of model instances
        is_fitted_ (bool): Whether the model has been fitted
    """

    SUPPORTED_MODELS = {
        "bradley_terry": BradleyTerry,
        "gssd": GSSD,
        "prp": PRP,
        "toor": TOOR,
        "zsd": ZSD,
        "poisson": Poisson,
    }

    def __init__(
        self,
        model_names: List[str] = list(SUPPORTED_MODELS.keys()),
    ):
        """Initialize ensemble model.

        Args:
            model_names: List of model names to include

        Raises:
            ValueError: If invalid model names are provided
        """
        # Validate model names
        invalid_models = set(model_names) - set(self.SUPPORTED_MODELS.keys())
        if invalid_models:
            raise ValueError(f"Unsupported models: {invalid_models}")

        # Initialize models with type safety
        self.models: Dict[str, Union[BradleyTerry, GSSD, PRP, TOOR, ZSD]] = {
            name: self.SUPPORTED_MODELS[name]() for name in model_names
        }

        self.is_fitted_ = False
        self.team_map_ = {}

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "BaseEnsemble":
        """Fit all models in the ensemble.

        Args:
            X: DataFrame containing match data
                Required columns:
                - 'home_team': Home team names
                - 'away_team': Away team names
                - 'goal_difference': Goal difference (home - away) if y is None
            y: Optional target values (goal differences if not in X)
            Z: Optional additional data (e.g., scores data)
                - 'home_goals': Goals scored by home team
                - 'away_goals': Goals scored by away team
            weights: Optional weights for optimization

        Returns:
            self: The fitted ensemble model
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
    ) -> np.ndarray:
        """Generate ensemble spread predictions.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            point_spread: Point spread adjustment

        Returns:
            Predicted point spreads

        Raises:
            ValueError: If models haven't been fitted
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
        """Generate ensemble probability predictions.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            point_spread: Point spread adjustment
            include_draw: Whether to include draw probability
            outcome: Optional[str], default=None
                Outcome to predict (home_win, draw, away_win)

        Returns:
            Array of probabilities [home_win_prob, draw_prob, away_win_prob]

        Raises:
            ValueError: If models haven't been fitted
        """
        pass

    def _validate_X(self, X: pd.DataFrame, fit: bool = True) -> None:
        """Validate input DataFrame dimensions and types.

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

    def _check_is_fitted(self) -> None:
        """Validate that the model has been fitted.

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.is_fitted_:
            raise ValueError("Models have not been fitted yet")

    def _validate_teams(self, teams: List[str]) -> None:
        """Validate teams exist in the model."""
        for team in teams:
            if team not in self.team_map_:
                raise ValueError(f"Unknown team: {team}")
