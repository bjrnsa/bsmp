# %%

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.base_model import BaseModel
from bsmp.sports_model.utils import dixon_coles_weights


class BradleyTerry(BaseModel):
    """
    Bradley-Terry model for predicting match outcomes with scikit-learn-like API.

    A probabilistic model that estimates team ratings and predicts match outcomes
    using maximum likelihood estimation. The model combines logistic regression for
    win probabilities with OLS regression for point spread predictions.

    Parameters
    ----------
    home_advantage : float, default=0.1
        Initial value for home advantage parameter.

    Attributes
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    n_teams_ : int
        Number of teams in the dataset
    team_map_ : Dict[str, int]
        Mapping of team names to indices
    params_ : np.ndarray
        Optimized model parameters after fitting
        [0:n_teams_] - Team ratings
        [-1] - Home advantage parameter
    intercept_ : float
        Point spread model intercept
    spread_coefficient_ : float
        Point spread model coefficient
    spread_error_ : float
        Standard error of spread predictions
    """

    NAME = "BT"

    def __init__(self, home_advantage: float = 0.1) -> None:
        """Initialize Bradley-Terry model."""
        self.home_advantage_ = home_advantage
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
    ) -> "BradleyTerry":
        """
        Fit the Bradley-Terry model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
            If y is None, X must have a third column with goal differences.
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Goal differences (home - away). If provided, this will be used instead of
            the third column in X.
        Z : Optional[pd.DataFrame], default=None
            Additional data for the model, such as home_goals and away_goals.
            No column name checking is performed, only dimension validation.
        ratings_weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        match_weights : Optional[np.ndarray], default=None
            Weights for spread prediction

        Returns
        -------
        self : BradleyTerry
            Fitted model
        """
        try:
            # Validate input dimensions and types
            self._validate_X(X)

            # Validate Z dimensions if provided
            if Z is not None and len(Z) != len(X):
                raise ValueError("Z must have the same number of rows as X")

            # Extract team data (first two columns)
            self.home_team_ = X.iloc[:, 0].to_numpy()
            self.away_team_ = X.iloc[:, 1].to_numpy()

            # Handle goal difference (y)
            if y is not None:
                self.goal_difference_ = np.asarray(y)
            elif X.shape[1] >= 3:
                self.goal_difference_ = X.iloc[:, 2].to_numpy()
            else:
                raise ValueError(
                    "Either y or a third column in X with goal differences must be provided"
                )

            # Validate goal difference
            if not np.issubdtype(self.goal_difference_.dtype, np.number):
                raise ValueError("Goal differences must be numeric")

            # Derive result from goal_difference
            self.result_ = np.zeros_like(self.goal_difference_, dtype=int)
            self.result_[self.goal_difference_ > 0] = 1
            self.result_[self.goal_difference_ < 0] = -1
            # result = 0 when goal_difference = 0 (draw)

            # Team setup
            self.teams_ = np.unique(np.concatenate([self.home_team_, self.away_team_]))
            self.n_teams_ = len(self.teams_)
            self.team_map_ = {team: idx for idx, team in enumerate(self.teams_)}

            # Create team indices
            self.home_idx_ = np.array(
                [self.team_map_[team] for team in self.home_team_]
            )
            self.away_idx_ = np.array(
                [self.team_map_[team] for team in self.away_team_]
            )

            # Set weights
            n_matches = len(X)
            self.ratings_weights_ = (
                np.ones(n_matches) if ratings_weights is None else ratings_weights
            )
            self.match_weights_ = (
                np.ones(n_matches) if match_weights is None else match_weights
            )

            # Initialize parameters
            self.params_ = np.zeros(self.n_teams_ + 1)
            self.params_[-1] = self.home_advantage_

            # Optimize parameters
            self.params_ = self._optimize_parameters()

            # Fit point spread model
            rating_diff = self._get_rating_difference()
            (self.intercept_, self.spread_coefficient_), self.spread_error_ = (
                self._fit_ols(self.goal_difference_, rating_diff)
            )

            self.is_fitted_ = True
            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
    ) -> np.ndarray:
        """
        Predict point spreads for matches.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        Z : Optional[pd.DataFrame], default=None
            Additional data for prediction. No column name checking is performed,
            only dimension validation.
        point_spread : float, default=0.0
            Point spread adjustment

        Returns
        -------
        np.ndarray
            Predicted point spreads (goal differences)
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Validate Z dimensions if provided
        if Z is not None and len(Z) != len(X):
            raise ValueError("Z must have the same number of rows as X")

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Calculate rating difference
            rating_diff = self._get_rating_difference(
                home_idx=self.team_map_[home_team],
                away_idx=self.team_map_[away_team],
            )

            # Calculate predicted spread
            predicted_spreads[i] = (
                self.intercept_ + self.spread_coefficient_ * rating_diff
            )

        return predicted_spreads

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> np.ndarray:
        """
        Predict match outcome probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        Z : Optional[pd.DataFrame], default=None
            Additional data for prediction. No column name checking is performed,
            only dimension validation.
        point_spread : float, default=0.0
            Point spread adjustment
        include_draw : bool, default=True
            Whether to include draw probability

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with probabilities
            If include_draw=True: [home_win, draw, away_win]
            If include_draw=False: [home_win, away_win]
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Validate Z dimensions if provided
        if Z is not None and len(Z) != len(X):
            raise ValueError("Z must have the same number of rows as X")

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        n_classes = 3 if include_draw else 2
        probabilities = np.zeros((len(X), n_classes))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Calculate rating difference
            rating_diff = self._get_rating_difference(
                home_idx=self.team_map_[home_team],
                away_idx=self.team_map_[away_team],
            )

            # Calculate predicted spread
            predicted_spread = self.intercept_ + self.spread_coefficient_ * rating_diff

            # Calculate probabilities
            if include_draw:
                thresholds = np.array([point_spread + 0.5, -point_spread - 0.5])
                probs = stats.norm.cdf(thresholds, predicted_spread, self.spread_error_)
                probabilities[i] = [1 - probs[0], probs[0] - probs[1], probs[1]]
            else:
                prob_home = 1 - stats.norm.cdf(
                    point_spread, predicted_spread, self.spread_error_
                )
                probabilities[i] = [prob_home, 1 - prob_home]

        return probabilities

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Calculate negative log likelihood for parameter optimization."""
        ratings = params[:-1]
        home_advantage = params[-1]
        log_likelihood = 0.0

        # Precompute home and away ratings
        home_ratings = ratings[self.home_idx_]
        away_ratings = ratings[self.away_idx_]
        win_probs = self._logit_transform(home_advantage + home_ratings - away_ratings)

        # Vectorized calculation
        win_mask = self.result_ == 1
        loss_mask = self.result_ == -1
        draw_mask = ~(win_mask | loss_mask)

        log_likelihood += np.sum(
            self.ratings_weights_[win_mask] * np.log(win_probs[win_mask])
        )
        log_likelihood += np.sum(
            self.ratings_weights_[loss_mask] * np.log(1 - win_probs[loss_mask])
        )
        log_likelihood += np.sum(
            self.ratings_weights_[draw_mask]
            * (np.log(win_probs[draw_mask]) + np.log(1 - win_probs[draw_mask]))
        )

        return -log_likelihood

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP."""
        result = minimize(
            fun=lambda p: self._log_likelihood(p) / len(self.result_),
            x0=self.params_,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )
        return result.x

    def _get_rating_difference(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
    ) -> np.ndarray:
        """Calculate rating difference between teams."""
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

        ratings = self.params_[:-1]
        home_advantage = self.params_[-1]
        return self._logit_transform(
            home_advantage + ratings[home_idx] - ratings[away_idx]
        )

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply logistic transformation."""
        return 1 / (1 + np.exp(-x))

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit weighted OLS regression using match weights."""
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights_)

        # Use more efficient matrix operations
        XtW = X.T @ W
        coefficients = np.linalg.solve(XtW @ X, XtW @ y)
        residuals = y - X @ coefficients
        weighted_sse = residuals.T @ W @ residuals
        std_error = np.sqrt(weighted_sse / (len(y) - X.shape[1]))

        return coefficients, std_error

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with team ratings
        """
        self._check_is_fitted()
        return pd.DataFrame(
            data=self.params_,
            index=list(self.teams_) + ["Home Advantage"],
            columns=["rating"],
        )

    def get_params(self) -> dict:
        """
        Get the current parameters of the model.

        Returns
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "home_advantage": self.home_advantage_,
            "params": self.params_,
            "is_fitted": self.is_fitted_,
        }

    def set_params(self, params: dict) -> None:
        """
        Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.home_advantage_ = params["home_advantage"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]


# %%
if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
    team_weights = dixon_coles_weights(train_df.datetime)

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = BradleyTerry()

    # Prepare training data
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    Z_train = train_df[["home_goals", "away_goals"]]
    model.fit(X_train, y_train, Z=Z_train)

    # Display team ratings
    print(model.get_team_ratings())

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
