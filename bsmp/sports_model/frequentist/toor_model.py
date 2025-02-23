# %%
from typing import Optional

import numpy as np
import pandas as pd

from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry
from bsmp.sports_model.utils import dixon_coles_weights


class TOOR(BradleyTerry):
    """
    Team OLS Optimized Rating (TOOR) model.

    An extension of the Bradley-Terry model that uses team-specific coefficients
    for more accurate point spread predictions. The model combines traditional
    Bradley-Terry ratings with a team-specific regression approach.

    Attributes:
        team_coefficients (np.ndarray): Team-specific regression coefficients
        home_coefficient (float): Home advantage coefficient
        home_team_coef (float): Home team rating coefficient
        away_team_coef (float): Away team rating coefficient
        spread_error (float): Standard error of spread predictions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
        home_advantage: float = 0.1,
    ):
        """
        Initialize TOOR model.

        Args:
            df: DataFrame with columns [home_team, away_team, result, goal_difference]
            ratings_weights: Optional weights for ratings optimization
            match_weights: Optional weights for match prediction
            home_advantage: Home field advantage parameter
        """
        super().__init__(df, ratings_weights, match_weights, home_advantage)
        self.team_coefficients = None
        self.home_coefficient = None
        self.home_team_coef = None
        self.away_team_coef = None

    def fit(self) -> None:
        """
        Fit the model in two steps:
        1. Calculate Bradley-Terry ratings
        2. Fit team-specific regression for point spread prediction
        """
        # Get team ratings from Bradley-Terry model
        self.best_params = self._optimize_parameters()
        self.team_ratings = dict(zip(self.teams, self.best_params[:-1]))

        # Prepare data for team-specific regression
        home_ratings = self._get_team_ratings(self.home_team)
        away_ratings = self._get_team_ratings(self.away_team)
        X = np.column_stack((home_ratings, away_ratings))

        # Fit regression model with team-specific coefficients
        coefficients, self.spread_error = self._fit_ols(self.goal_difference, X)
        self.home_coefficient, self.home_team_coef, self.away_team_coef = coefficients

    def predict(
        self,
        home_team: str,
        away_team: str,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> tuple:
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
        # Get team ratings
        home_rating = self.team_ratings[home_team]
        away_rating = self.team_ratings[away_team]

        # Calculate predicted spread using team-specific coefficients
        predicted_spread = (
            self.home_coefficient
            + self.home_team_coef * home_rating
            + self.away_team_coef * away_rating
        )

        return self._calculate_probabilities(
            predicted_spread, self.spread_error, point_spread, include_draw
        )


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

    home_team = "Richmond"
    away_team = "Geelong"

    model = TOOR(df, ratings_weights=team_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=15, include_draw=False
    )

    # Use different weights
    model = TOOR(df, ratings_weights=team_weights, match_weights=spread_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=15, include_draw=False
    )
    # Use no weights
    model = TOOR(df)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=15, include_draw=False
    )
