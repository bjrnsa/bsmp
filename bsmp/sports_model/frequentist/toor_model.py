# %%
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry
from bsmp.sports_model.utils import dixon_coles_weights


class TOOR(BradleyTerry):
    """
    Team OLS Optimized Rating (TOOR) model.

    An extension of the Bradley-Terry model that uses team-specific coefficients
    for more accurate point spread predictions. The model combines traditional
    Bradley-Terry ratings with a team-specific regression approach.

    Attributes:
        Inherits all attributes from BradleyTerry plus:
        team_coefficients (np.ndarray): Team-specific regression coefficients
        home_coefficient (float): Home advantage coefficient
        home_team_coef (float): Home team rating coefficient
        away_team_coef (float): Away team rating coefficient
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
        home_advantage: float = 0.1,
    ):
        """Initialize TOOR model."""
        super().__init__(df, ratings_weights, match_weights, home_advantage)
        self.team_coefficients = None
        self.home_coefficient = None
        self.home_team_coef = None
        self.away_team_coef = None

    def fit(self) -> "TOOR":
        """
        Fit the model in two steps:
        1. Calculate Bradley-Terry ratings
        2. Fit team-specific regression for point spread prediction
        """
        try:
            # Get team ratings from Bradley-Terry model
            self.params = self._optimize_parameters()

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
            self.home_coefficient = result.x[0]  # home advantage
            self.home_team_coef = result.x[1]  # home team coefficient
            self.away_team_coef = result.x[2]  # away team coefficient

            # Calculate spread error
            predictions = (
                self.home_coefficient
                + self.home_team_coef * self.params[self.home_idx]
                + self.away_team_coef * self.params[self.away_idx]
            )
            (self.intercept, self.spread_coefficient), self.spread_error = (
                self._fit_ols(self.goal_difference, predictions)
            )

            self.fitted = True
            return self

        except Exception as e:
            self.fitted = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def predict(
        self,
        home_team: str,
        away_team: str,
        point_spread: float = 0.0,
        include_draw: bool = True,
        return_spread: bool = False,
    ) -> Union[Tuple[float, float, float], float]:
        """
        Predict match outcome probabilities using team-specific coefficients.

        Args:
            home_team: Name of home team
            away_team: Name of away team
            point_spread: Point spread adjustment
            include_draw: Whether to include draw probability

        Returns:
            tuple: (home_win_prob, draw_prob, away_win_prob)
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        # Validate teams
        for team in (home_team, away_team):
            if team not in self.team_map:
                raise ValueError(f"Unknown team: {team}")

        # Get team ratings
        home_rating = self.params[self.team_map[home_team]]
        away_rating = self.params[self.team_map[away_team]]

        # Calculate predicted spread using team-specific coefficients
        predicted_spread = (
            self.home_coefficient
            + self.home_team_coef * home_rating
            + self.away_team_coef * away_rating
        )
        if return_spread:
            return predicted_spread

        return self._calculate_probabilities(
            predicted_spread, self.spread_error, point_spread, include_draw
        )

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP."""
        result = minimize(
            fun=lambda p: self._log_likelihood(p) / len(self.result),
            x0=self.params,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )
        return result.x

    def _sse_function(self, parameters: np.ndarray) -> float:
        """
        Calculate sum of squared errors for parameter optimization.

        Args:
            parameters: Array of [home_advantage, home_team_coef, away_team_coef]

        Returns:
            float: Sum of squared errors
        """
        home_adv, home_team_coef, away_team_coef = parameters

        # Get logistic ratings from Bradley-Terry optimization
        logistic_ratings = self.params[:-1]  # Exclude home advantage parameter

        # Calculate predictions
        predictions = (
            home_adv
            + home_team_coef * logistic_ratings[self.home_idx]
            + away_team_coef * logistic_ratings[self.away_idx]
        )

        # Calculate weighted squared errors
        errors = self.goal_difference - predictions
        if hasattr(self, "match_weights"):
            sse = np.sum(self.match_weights * (errors**2))
        else:
            sse = np.sum(errors**2)

        return sse


if __name__ == "__main__":
    # Load AFL data for testing
    df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv").loc[:176]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["game_total"] = df["away_pts"] + df["home_pts"]
    df["goal_difference"] = df["home_pts"] - df["away_pts"]
    df["result"] = np.where(df["goal_difference"] > 0, 1, -1)

    df["date"] = pd.to_datetime(df["date"], format="%b %d (%a %I:%M%p)")
    team_weights = dixon_coles_weights(df.date)
    np.random.seed(0)
    spread_weights = np.random.uniform(0.1, 1.0, len(df))

    home_team = "St Kilda"
    away_team = "North Melbourne"

    # Test with different weight configurations
    model = TOOR(df)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )
