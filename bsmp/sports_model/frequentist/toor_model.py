# %%
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry
from bsmp.sports_model.utils import dixon_coles_weights


class TOOR(BradleyTerry):
    """
    Team OLS Optimized Rating (TOOR) model with scikit-learn-like API.

    An extension of the Bradley-Terry model that uses team-specific coefficients
    for more accurate point spread predictions. The model combines traditional
    Bradley-Terry ratings with a team-specific regression approach.

    Parameters
    ----------
    home_advantage : float, default=0.1
        Initial value for home advantage parameter.

    Attributes
    ----------
    Inherits all attributes from BradleyTerry plus:
    home_coefficient_ : float
        Home advantage coefficient for spread prediction
    home_team_coef_ : float
        Home team rating coefficient
    away_team_coef_ : float
        Away team rating coefficient
    """

    NAME = "TOOR"

    def __init__(self, home_advantage: float = 0.1) -> None:
        """Initialize TOOR model."""
        super().__init__(home_advantage=home_advantage)
        self.home_coefficient_ = None
        self.home_team_coef_ = None
        self.away_team_coef_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
    ) -> "TOOR":
        """
        Fit the TOOR model in two steps:
        1. Calculate Bradley-Terry ratings
        2. Fit team-specific regression for point spread prediction

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
        self : TOOR
            Fitted model
        """
        try:
            # First fit the Bradley-Terry model to get team ratings
            super().fit(X, y, Z, ratings_weights, match_weights)

            # Optimize the three parameters using least squares
            initial_guess = np.array(
                [0.1, 1.0, -1.0]
            )  # Initial values for [home_adv, home_team, away_team]
            result = minimize(
                self._sse_function,
                initial_guess,
                method="L-BFGS-B",
                options={"ftol": 1e-10, "maxiter": 200},
            )

            # Store the optimized coefficients
            self.home_coefficient_ = result.x[0]  # home advantage
            self.home_team_coef_ = result.x[1]  # home team coefficient
            self.away_team_coef_ = result.x[2]  # away team coefficient

            # Calculate spread error
            predictions = (
                self.home_coefficient_
                + self.home_team_coef_ * self.params_[self.home_idx_]
                + self.away_team_coef_ * self.params_[self.away_idx_]
            )
            (self.intercept_, self.spread_coefficient_), self.spread_error_ = (
                self._fit_ols(self.goal_difference_, predictions)
            )

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
        Predict point spreads for matches using team-specific coefficients.

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

            # Get team ratings
            home_rating = self.params_[self.team_map_[home_team]]
            away_rating = self.params_[self.team_map_[away_team]]

            # Calculate predicted spread using team-specific coefficients
            predicted_spreads[i] = (
                self.home_coefficient_
                + self.home_team_coef_ * home_rating
                + self.away_team_coef_ * away_rating
            ) * self.spread_coefficient_ + self.intercept_

        return predicted_spreads

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> np.ndarray:
        """
        Predict match outcome probabilities using team-specific coefficients.

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

            # Get team ratings
            home_rating = self.params_[self.team_map_[home_team]]
            away_rating = self.params_[self.team_map_[away_team]]

            # Calculate predicted spread using team-specific coefficients
            predicted_spread = (
                self.home_coefficient_
                + self.home_team_coef_ * home_rating
                + self.away_team_coef_ * away_rating
            ) * self.spread_coefficient_ + self.intercept_

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

    def _sse_function(self, parameters: np.ndarray) -> float:
        """
        Calculate sum of squared errors for parameter optimization.

        Parameters
        ----------
        parameters : np.ndarray
            Array of [home_advantage, home_team_coef, away_team_coef]

        Returns
        -------
        float
            Sum of squared errors
        """
        home_adv, home_team_coef, away_team_coef = parameters

        # Get logistic ratings from Bradley-Terry optimization
        logistic_ratings = self.params_[:-1]  # Exclude home advantage parameter

        # Calculate predictions
        predictions = (
            home_adv
            + home_team_coef * logistic_ratings[self.home_idx_]
            + away_team_coef * logistic_ratings[self.away_idx_]
        )

        # Calculate weighted squared errors
        errors = self.goal_difference_ - predictions
        sse = np.sum(self.match_weights_ * (errors**2))

        return sse


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

    # train_df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv").iloc[:176]
    # train_df = train_df.assign(
    #     goal_difference=train_df["Home Pts"] - train_df["Away Pts"]
    # )
    # train_df = train_df.assign(
    #     home_team=train_df["Home Team"], away_team=train_df["Away Team"]
    # )
    # home_team = "St Kilda"
    # away_team = "North Melbourne"

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = TOOR()

    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    model.fit(X_train, y_train)
    model.get_team_ratings()

    X_test = test_df[["home_team", "away_team"]]
    y_test = test_df["goal_difference"]

    # Predict point spreads (goal differences)
    predicted_spread = model.predict(X_test)

    # Predict probabilities
    probs = model.predict_proba(X_test, point_spread=0, include_draw=True)

    print(f"Home win probability: {probs[0, 0]:.4f}")
    print(f"Draw probability: {probs[0, 1]:.4f}")
    print(f"Away win probability: {probs[0, 2]:.4f}")
    print(f"Home rating: {model.get_team_rating(home_team):.4f}")
    print(f"Away rating: {model.get_team_rating(away_team):.4f}")

# %%
