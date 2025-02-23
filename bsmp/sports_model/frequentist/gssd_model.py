# %%
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from bsmp.sports_model.utils import dixon_coles_weights


class GSSD:
    """
    Generalized Scores Standard Deviation (GSSD) model.

    A model that predicts match outcomes using team-specific offensive and defensive ratings.
    The model uses weighted OLS regression to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

    Attributes:
        teams (np.ndarray): Unique team identifiers
        team_ratings (dict): Dictionary mapping teams to their offensive/defensive ratings
        is_fitted (bool): Whether the model has been fitted
        std_error (float): Standard error of the model predictions
        intercept (float): Model intercept term
        pfh_coeff (float): Coefficient for home team's offensive rating
        pah_coeff (float): Coefficient for home team's defensive rating
        pfa_coeff (float): Coefficient for away team's offensive rating
        paa_coeff (float): Coefficient for away team's defensive rating
    """

    def __init__(self, df: pd.DataFrame, match_weights: Union[np.ndarray, None] = None):
        """
        Initialize GSSD model.

        Args:
            df (pd.DataFrame): DataFrame containing match data with columns:
                - home_team: Home team identifier
                - away_team: Away team identifier
                - home_pts: Points scored by home team
                - away_pts: Points scored by away team
                - goal_difference: home_pts - away_pts
            match_weights (np.ndarray, optional): Weights for match prediction.
                If None, equal weights are used.
        """
        # Basic setup
        self.teams = np.unique(df.home_team.to_numpy())
        self.home_team = df.home_team.to_numpy()
        self.goal_difference = df.goal_difference.to_numpy()
        self.match_weights = (
            np.ones(len(df)) if match_weights is None else match_weights
        )

        # Calculate team statistics
        self._calculate_team_statistics(df)
        self.fitted = False

    def fit(self):
        """
        Fit the GSSD model to the data.

        Estimates team-specific coefficients using weighted OLS regression.
        Sets is_fitted to True upon completion.
        """
        # Prepare features and fit model
        features = np.column_stack((self.pfh, self.pah, self.pfa, self.paa))
        model_params, self.std_error = self._fit_ols(self.goal_difference, features)

        # Store model parameters
        (
            self.intercept,
            self.pfh_coeff,
            self.pah_coeff,
            self.pfa_coeff,
            self.paa_coeff,
        ) = model_params

        self.fitted = True

    def predict(
        self,
        home_team: str,
        away_team: str,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Predict match outcome probabilities.

        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            point_spread (float, optional): Point spread adjustment. Defaults to 0.0
            include_draw (bool, optional): Whether to include draw probability.
                Defaults to True

        Returns:
            Tuple[float, float, float]: Probabilities of (home win, draw, away win)
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        # Get team ratings
        home_off, home_def = self.team_ratings[home_team][:2]
        away_off, away_def = self.team_ratings[away_team][2:]

        # Calculate spread
        predicted_spread = (
            self.intercept
            + home_off * self.pfh_coeff
            + home_def * self.pah_coeff
            + away_off * self.pfa_coeff
            + away_def * self.paa_coeff
        )

        return self._calculate_probabilities(
            predicted_spread=predicted_spread,
            std_error=self.std_error,
            point_spread=point_spread,
            include_draw=include_draw,
        )

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit weighted OLS regression.

        Args:
            y (np.ndarray): Target variable (goal differences)
            X (np.ndarray): Feature matrix of team statistics

        Returns:
            Tuple[np.ndarray, float]: Model coefficients and standard error
        """
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights)

        coefficients = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        residuals = y - X @ coefficients
        weighted_sse = residuals.T @ W @ residuals
        std_error = np.sqrt(weighted_sse / (len(y) - X.shape[1]))

        return coefficients, std_error

    def _calculate_probabilities(
        self,
        predicted_spread: float,
        std_error: float,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Calculate win/draw/loss probabilities using normal distribution.

        Args:
            predicted_spread (float): Predicted point spread
            std_error (float): Standard error of the prediction
            point_spread (float, optional): Point spread adjustment. Defaults to 0.0
            include_draw (bool, optional): Whether to include draw probability.
                Defaults to True

        Returns:
            Tuple[float, float, float]: Probabilities of (home win, draw, away win)
        """
        if include_draw:
            prob_home = 1 - stats.norm.cdf(
                point_spread + 0.5, predicted_spread, std_error
            )
            prob_away = stats.norm.cdf(-point_spread - 0.5, predicted_spread, std_error)
            prob_draw = 1 - prob_home - prob_away
        else:
            prob_home = 1 - stats.norm.cdf(point_spread, predicted_spread, std_error)
            prob_draw = np.nan
            prob_away = stats.norm.cdf(-point_spread, predicted_spread, std_error)

        return prob_home, prob_draw, prob_away

    def _calculate_team_statistics(self, df: pd.DataFrame) -> None:
        """
        Calculate and store all team-related statistics.

        Args:
            df (pd.DataFrame): DataFrame containing match data
        """
        # Calculate mean points for home/away scenarios
        home_stats = df.groupby("home_team").agg(
            {"home_pts": "mean", "away_pts": "mean"}
        )
        away_stats = df.groupby("away_team").agg(
            {"away_pts": "mean", "home_pts": "mean"}
        )

        # Store transformed statistics
        self.pfh = df.groupby("home_team")["home_pts"].transform("mean").to_numpy()
        self.pah = df.groupby("home_team")["away_pts"].transform("mean").to_numpy()
        self.pfa = df.groupby("away_team")["away_pts"].transform("mean").to_numpy()
        self.paa = df.groupby("away_team")["home_pts"].transform("mean").to_numpy()

        # Create team ratings dictionary
        self.team_ratings = {
            team: np.array(
                [
                    home_stats.loc[team, "home_pts"],
                    home_stats.loc[team, "away_pts"],
                    away_stats.loc[team, "away_pts"],
                    away_stats.loc[team, "home_pts"],
                ]
            )
            for team in self.teams
        }


if __name__ == "__main__":
    # Load AFL data for testing
    df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv")  # .loc[:176]
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

    # Use no weights
    model = GSSD(df)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )

    model = GSSD(df, match_weights=team_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )
    # Use different weights
    model = GSSD(df, match_weights=spread_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )
