# %%
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry
from bsmp.sports_model.frequentist.gssd_model import GSSD
from bsmp.sports_model.frequentist.prp_model import PRP
from bsmp.sports_model.frequentist.toor_model import TOOR
from bsmp.sports_model.frequentist.zsd_model import ZSD
from bsmp.sports_model.utils import dixon_coles_weights


class Ensemble:
    """
    Ensemble model that combines predictions from multiple sports prediction models.

    The ensemble uses simple averaging or more advanced methods like NNLS for spreads
    and logistic regression for probabilities to create more robust predictions.

    Supported models include:
    - Bradley-Terry
    - GSSD (Generalized Scores Standard Deviation)
    - PRP (Points Rating Prediction)
    - TOOR (Team OLS Optimized Rating)
    - ZSD (Z-Score Deviation)

    Attributes:
        models (Dict[str, Union[BradleyTerry, GSSD, PRP, TOOR, ZSD]]): Dictionary of model instances
        is_fitted_ (bool): Whether the model has been fitted
        spread_ensemble_ (Optional[LinearRegression]): NNLS model for ensemble weighting of spreads
        proba_ensemble_ (Optional[LogisticRegression]): Logistic regression model for ensemble weighting of probabilities
    """

    SUPPORTED_MODELS = {
        "bradley_terry": BradleyTerry,
        "gssd": GSSD,
        "prp": PRP,
        "toor": TOOR,
        "zsd": ZSD,
    }

    def __init__(
        self,
        model_names: List[str],
    ):
        """
        Initialize ensemble model.

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
        self.spread_ensemble_ = None
        self.proba_ensemble_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
    ) -> "Ensemble":
        """
        Fit all models in the ensemble.

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
            ratings_weights: Optional weights for rating optimization
            match_weights: Optional weights for spread prediction

        Returns:
            self: The fitted ensemble model
        """
        try:
            # Validate that X has the required columns
            required_cols = ["home_team", "away_team"]
            if y is None:
                required_cols.append("goal_difference")

            missing_cols = [col for col in required_cols if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in X: {missing_cols}")

            # Prepare data for each model based on its requirements
            for name, model in self.models.items():
                print(f"Fitting {name} model...")
                model.fit(X, y, Z, ratings_weights, match_weights)

            self.is_fitted_ = True
            # Fit ensemble
            self.fit_spread_ensemble(X, y, Z)

            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Ensemble fitting failed: {str(e)}") from e

    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        ensemble_method: str = "simple_average",
    ) -> np.ndarray:
        """
        Generate ensemble spread predictions by averaging individual model predictions.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            point_spread: Point spread adjustment
            ensemble_method: Method to combine model predictions
                - "simple_average": Simple average of model predictions
                - "nnls": Weighted average using NNLS

        Returns:
            Predicted point spreads

        Raises:
            ValueError: If models haven't been fitted
        """
        self._check_is_fitted()

        # Validate that X has the required columns
        required_cols = ["home_team", "away_team"]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in X: {missing_cols}")

        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            model_preds = model.predict(X, Z, point_spread=point_spread)
            predictions.append(model_preds)

        # Stack predictions
        predictions = np.column_stack(predictions)

        # If NNLS ensemble is fitted, use it; otherwise use simple average
        if ensemble_method == "nnls":
            if self.spread_ensemble_ is None:
                raise ValueError("NNLS ensemble not fitted")
            return self.spread_ensemble_.predict(predictions)
        else:
            return np.mean(predictions, axis=1)

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> np.ndarray:
        """
        Generate ensemble probability predictions.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)
            point_spread: Point spread adjustment
            include_draw: Whether to include draw probability

        Returns:
            Array of probabilities [home_win_prob, draw_prob, away_win_prob]

        Raises:
            ValueError: If models haven't been fitted
        """
        self._check_is_fitted()

        # Get probability predictions from all models
        all_probas = []
        for name, model in self.models.items():
            probas = model.predict_proba(
                X, Z, point_spread=point_spread, include_draw=include_draw
            )
            all_probas.append(probas)

        # Stack probabilities from all models
        stacked_probas = np.dstack(
            all_probas
        )  # Shape: (n_samples, n_outcomes, n_models)

        return np.mean(stacked_probas, axis=2)

    def fit_spread_ensemble(
        self, X: pd.DataFrame, y: np.ndarray, Z: Optional[pd.DataFrame] = None
    ) -> "Ensemble":
        """
        Fit NNLS ensemble for spread predictions.

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
        for name, model in self.models.items():
            model_preds = model.predict(X, Z)
            predictions.append(model_preds)

        # Stack predictions
        predictions = np.column_stack(predictions)

        # Fit NNLS ensemble
        self.spread_ensemble_ = LinearRegression(positive=True, fit_intercept=True)
        self.spread_ensemble_.fit(predictions, y)

        # Print coefficients for transparency
        coeffs = self.spread_ensemble_.coef_ / np.sum(self.spread_ensemble_.coef_)
        print("Spread Model Coefficients:")
        for name, coef in zip(self.models.keys(), coeffs):
            print(f"{name}: {coef:.3f}")

        return self

    def get_model_spreads(
        self, X: pd.DataFrame, Z: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get individual spread predictions from each model.

        Args:
            X: DataFrame containing match data with home_team and away_team columns
            Z: Optional additional data (e.g., scores data)

        Returns:
            DataFrame with spread predictions from each model
        """
        self._check_is_fitted()

        # Get predictions from all models
        spreads = {}
        for name, model in self.models.items():
            spreads[name] = model.predict(X, Z)

        spreads["ens"] = self.predict(X, Z, ensemble_method="simple_average")
        spreads["nnls"] = self.predict(X, Z, ensemble_method="nnls")

        return pd.DataFrame(spreads, index=X.index)

    def get_model_probabilities(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        include_draw: bool = True,
    ) -> pd.DataFrame:
        """
        Get individual probability predictions from each model.

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
        outcomes = ["home", "draw", "away"] if include_draw else ["home", "away"]
        index = pd.MultiIndex.from_product(
            [X.index, outcomes], names=["match", "outcome"]
        )

        # Reshape and combine probabilities
        result = pd.DataFrame(index=index)
        for name, probs in probabilities.items():
            flat_probs = probs.reshape(-1)
            result[name] = flat_probs

        return result

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings from all models.

        Returns:
            DataFrame with team ratings from each model
        """
        self._check_is_fitted()

        ratings = {}
        for name, model in self.models.items():
            try:
                model_ratings = model.get_team_ratings()
                ratings[name] = model_ratings["rating"]
            except (AttributeError, KeyError):
                continue

        return pd.DataFrame(ratings)

    def _check_is_fitted(self) -> None:
        """
        Validate that the model has been fitted.

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.is_fitted_:
            raise ValueError("Models have not been fitted yet")


# %%
if __name__ == "__main__":
    """
    Example of using the Ensemble class for sports prediction.
    """
    # Load example data (using AFL data from the frequentist directory)
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )

    weights = dixon_coles_weights(train_df.datetime, xi=0.018)
    # Create Z dataframe with score data
    Z_train = (
        train_df[["home_goals", "away_goals"]]
        if "home_goals" in train_df.columns
        else None
    )
    Z_test = (
        test_df[["home_goals", "away_goals"]]
        if "home_goals" in test_df.columns
        else None
    )

    # Split data into train and test sets
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]

    X_test = test_df[["home_team", "away_team"]]
    y_test = test_df["goal_difference"]

    # Create ensemble with all available models
    model_names = ["bradley_terry", "gssd", "prp", "toor", "zsd"]
    ensemble = Ensemble(model_names)

    # Fit the ensemble
    print("\nFitting base models...")
    ensemble.fit(X_train, y_train, Z_train, ratings_weights=weights)

    # Get individual model spreads
    model_spreads = ensemble.get_model_spreads(X_test)

    results_df = pd.concat([y_test, model_spreads], axis=1)

    # Create a DataFrame to hold the metrics
    metrics_df = pd.DataFrame(
        {
            **{
                model: [
                    np.mean(
                        np.abs(results_df[y_test.name] - model_spreads[model].values)
                    ),
                    np.mean(
                        (results_df[y_test.name] - model_spreads[model].values) ** 2
                    ),
                ]
                for model in model_spreads.columns
            },
        },
        index=["MAE", "MSE"],
    )

    results_df.plot.scatter(x="nnls", y="goal_difference")
    results_df.describe()
    # %%
    # Make probability predictions

    # Get individual model probabilities
    model_probas = ensemble.get_model_probabilities(X_test, Z_test)


# %%
