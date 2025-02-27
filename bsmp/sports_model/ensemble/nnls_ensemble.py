# %%
"""This file contains the implementation of the Non-Negative Least Squares (NNLS) ensemble model for predicting sports match outcomes."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.ensemble.base_ensemble import BaseEnsemble
from bsmp.sports_model.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry, Poisson
from bsmp.sports_model.utils import dixon_coles_weights


class NNLS(BaseEnsemble):
    """NNLS ensemble model that combines predictions from multiple sports models.

    The ensemble uses NNLS for spreads to create more robust predictions by learning
    optimal weights for each model's predictions.

    Attributes:
        models (Dict[str, Union[BradleyTerry, GSSD, PRP, TOOR, ZSD]]): Dictionary of model instances
        is_fitted_ (bool): Whether the model has been fitted
        spread_ensemble_ (Optional[LinearRegression]): NNLS model for ensemble weighting of spreads
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
        super().__init__(model_names)
        self.spread_ensemble_ = LinearRegression(positive=True, fit_intercept=True)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "NNLS":
        """Fit all models in the ensemble and learn optimal weights.

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
        try:
            # Validate that X has the required columns
            self._validate_X(X, fit=True)
            required_cols = ["home_team", "away_team"]
            if y is None:
                required_cols.append("goal_difference")
                y = X.iloc[:, 2].to_numpy()

            assert isinstance(y, (np.ndarray, pd.Series)), (
                "y must be a numpy array or pandas Series"
            )

            missing_cols = [col for col in required_cols if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in X: {missing_cols}")

            # Prepare data for each model based on its requirements
            for _, model in self.models.items():
                model.fit(X, y, Z, weights)

            self.is_fitted_ = True
            # Fit ensemble weights
            self._fit_spread_ensemble(X, y, Z)

            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Ensemble fitting failed: {str(e)}") from e

    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
    ) -> np.ndarray:
        """Generate ensemble spread predictions using learned weights.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            point_spread: Point spread adjustment

        Returns:
            Predicted point spreads

        Raises:
            ValueError: If models haven't been fitted
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Get predictions from all models
        predictions = []
        for _, model in self.models.items():
            model_preds = model.predict(X, Z, point_spread=point_spread)
            predictions.append(model_preds)

        # Stack predictions
        predictions = np.column_stack(predictions)

        return self.spread_ensemble_.predict(predictions)

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
    ) -> np.ndarray:
        """Generate ensemble probability predictions using simple averaging.

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
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Get probability predictions from all models
        all_probas = []
        for _, model in self.models.items():
            probas = model.predict_proba(
                X,
                Z,
                point_spread=point_spread,
                include_draw=include_draw,
                outcome=outcome,
            )
            all_probas.append(probas)

        if outcome:
            return np.mean(all_probas, axis=0).reshape(-1)
        # Stack probabilities from all models and take mean
        stacked_probas = np.dstack(all_probas)
        return np.mean(stacked_probas, axis=2)

    def _fit_spread_ensemble(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
    ) -> "NNLS":
        """Fit NNLS ensemble for spread predictions.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            y: Actual goal differences
            Z: Optional additional data (e.g., scores data)

        Returns:
            self: The fitted ensemble model
        """
        self._check_is_fitted()

        # Get predictions from all models
        predictions = []
        for _, model in self.models.items():
            model_preds = model.predict(X, Z)
            predictions.append(model_preds)

        # Stack predictions
        predictions = np.column_stack(predictions)

        # Fit NNLS ensemble
        self.spread_ensemble_ = LinearRegression(positive=True, fit_intercept=True)
        y_array = np.asarray(y)
        self.spread_ensemble_.fit(predictions, y_array)

        return self

    def get_model_probabilities(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        include_draw: bool = True,
    ) -> pd.DataFrame:
        """Get individual probability predictions from each model.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            include_draw: Whether to include draw probability

        Returns:
            DataFrame with probability predictions from each model
        """
        self._check_is_fitted()

        # Get probability predictions from all models
        probabilities = {}
        for name, model in self.models.items():
            probabilities[name] = model.predict_proba(X, Z, include_draw=include_draw)

        # Get ensemle probability
        probabilities["ens"] = self.predict_proba(X, Z, include_draw=include_draw)

        # Create multi-index DataFrame
        outcomes = (
            ["home_prob", "draw_prob", "away_prob"]
            if include_draw
            else ["home_prob", "away_prob"]
        )
        index = pd.MultiIndex.from_product(
            [X.index, outcomes], names=["match", "outcome"]
        )

        # Reshape and combine probabilities
        result = pd.DataFrame(index=index)
        for name, probs in probabilities.items():
            flat_probs = probs.reshape(-1)
            result[name] = flat_probs

        return result

    def get_ensemble_weights(self) -> pd.DataFrame:
        """Get the normalized (sum to one) weights used in the ensemble."""
        self._check_is_fitted()

        return pd.DataFrame(
            self.spread_ensemble_.coef_, index=list(self.models.keys())
        ).div(self.spread_ensemble_.coef_.sum())


# %%
if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        # seasons=["2024/2025"],
    )
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
    weights = dixon_coles_weights(train_df.datetime, xi=0.0018)

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = NNLS(model_names=["bradley_terry", "gssd", "prp", "toor", "zsd"])

    # Prepare training data
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    Z_train = train_df[["home_goals", "away_goals"]]
    model.fit(X_train, y_train, Z=Z_train, weights=weights)

    # Create a test DataFrame for prediction
    X_test = test_df[["home_team", "away_team"]]

    # Predict point spreads (goal differences)
    predicted_spreads = model.predict(X_test)
    print(f"Predicted goal difference: {predicted_spreads[0]:.2f}")

    # Predict probabilities
    probs = model.predict_proba(X_test, point_spread=0, include_draw=True)

    print(f"Home win probability: {probs[0, 0]:.4f}")
    print(f"Draw probability: {probs[0, 1]:.4f}")
    print(f"Away win probability: {probs[0, 2]:.4f}")
