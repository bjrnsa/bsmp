# %%

import numpy as np
import scipy.stats as stats
from scipy.optimize import approx_fprime, minimize

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.utils import dixon_coles_weights


class BradleyTerry:
    """
    Implementation of the Bradley-Terry model for predicting match outcomes.
    The model estimates team ratings and home field advantage to predict
    win probabilities and point spreads.
    """

    def __init__(self, df, weights=None, home_advantage: float = 0.1):
        """
        Initialize Bradley-Terry model.

        Args:
            df: DataFrame containing match data
            weights: Optional weights for each observation
            home_advantage: Initial home field advantage parameter
        """
        # Convert DataFrame columns to numpy arrays
        for col in df:
            setattr(self, col, df[col].to_numpy())

        # Initialize model parameters
        self.teams = np.unique(self.home_team)
        self.team_ratings = dict.fromkeys(self.teams, 1.0)
        self.params = np.array(list(self.team_ratings.values()) + [home_advantage])

        # Prepare training data
        self.data = np.array([self.home_team, self.away_team, self.result]).T
        self.n_observations = len(self.data)
        self.weights = np.ones(self.n_observations) if weights is None else weights

    def fit(self) -> None:
        """
        Fit the Bradley-Terry model to estimate team ratings and predict point spreads.
        """
        # Optimize model parameters
        self.best_params = self._optimize_parameters()
        self.team_ratings = dict(zip(self.teams, self.best_params[:-1]))
        self.home_advantage = self.best_params[-1]

        # Fit point spread model
        home_ratings = self._get_team_ratings(self.home_team)
        away_ratings = self._get_team_ratings(self.away_team)
        rating_diff = self._logit_transform(
            self.home_advantage + home_ratings - away_ratings
        )

        # Estimate spread parameters using OLS
        (self.intercept, self.spread_coefficient), self.spread_error = self._fit_ols(
            self.goal_difference, rating_diff
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> tuple:
        """
        Predict match outcome probabilities.

        Args:
            home_team: Name of home team
            away_team: Name of away team
            point_spread: Point spread adjustment
            include_draw: Whether to include draw probability

        Returns:
            tuple: (home_win_prob, draw_prob, away_win_prob)
        """
        rating_diff = 1 - self._calculate_rating_difference(home_team, away_team)
        predicted_spread = self.intercept + self.spread_coefficient * rating_diff

        return self._calculate_probabilities(
            predicted_spread, self.spread_error, point_spread, include_draw
        )

    def _optimize_parameters(self) -> np.ndarray:
        """
        Optimize model parameters using the SLSQP algorithm.
        """
        result = minimize(
            fun=lambda p: self._log_likelihood(p, self.data, self.weights)
            / self.n_observations,
            x0=self.params,
            jac=lambda p: approx_fprime(
                p,
                lambda x: self._log_likelihood(x, self.data, self.weights)
                / self.n_observations,
                epsilon=6.5e-07,
            ),
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )
        return result.x

    def _calculate_probabilities(
        self,
        predicted_spread: float,
        std_error: float,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> tuple:
        """Calculate win/draw/loss probabilities using normal distribution."""
        if include_draw:
            prob_home = 1 - stats.norm.cdf(
                point_spread + 0.5, predicted_spread, std_error
            )
            prob_away = 1 - stats.norm.cdf(
                -point_spread - 0.5, -predicted_spread, std_error
            )
            prob_draw = 1 - prob_home - prob_away
        else:
            prob_home = 1 - stats.norm.cdf(point_spread, predicted_spread, std_error)
            prob_draw = np.nan
            prob_away = 1 - stats.norm.cdf(-point_spread, -predicted_spread, std_error)

        return prob_home, prob_draw, prob_away

    def _log_likelihood(
        self, params: np.ndarray, data: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate negative log likelihood for parameter optimization."""
        ratings = params[:-1]
        home_advantage = params[-1]
        log_likelihood = 0.0

        for i, (home, away, result) in enumerate(data):
            home_rating = ratings[self.teams == home][0]
            away_rating = ratings[self.teams == away][0]
            win_prob = self._logit_transform(home_advantage + home_rating - away_rating)

            if result == 1:
                outcome_prob = win_prob
            elif result == -1:
                outcome_prob = 1 - win_prob
            else:  # Draw
                log_likelihood += weights[i] * (np.log(win_prob) + np.log(1 - win_prob))
                continue

            log_likelihood += weights[i] * np.log(outcome_prob)

        return -log_likelihood

    def _logit_transform(self, x: float) -> float:
        """Apply logistic transformation."""
        return 1 / (1 + np.exp(-x))

    def _calculate_rating_difference(self, home_team: str, away_team: str) -> float:
        """Calculate logit-transformed rating difference between teams."""
        return self._logit_transform(
            -(
                self.home_advantage
                + self.team_ratings[home_team]
                - self.team_ratings[away_team]
            )
        )

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> tuple:
        """Fit OLS regression and return parameters and standard error."""
        X = np.column_stack((np.ones(len(X)), X))
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        residuals = y - X @ coefficients
        std_error = np.sqrt(np.sum(residuals**2) / (len(y) - X.shape[1]))
        return coefficients, std_error

    def _get_team_ratings(self, teams: np.ndarray) -> np.ndarray:
        """Get ratings for a list of teams."""
        return np.array([self.team_ratings[team] for team in teams])


if __name__ == "__main__":
    # Initialize and fit the model
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(league="Herre Handbold Ligaen")
    odds_df = loader.load_odds(df.index.unique())
    weights = dixon_coles_weights(df.datetime, xi=0.18)

    import pandas as pd

    # Load AFL data for testing
    df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv").loc[:176]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["game_total"] = df["away_pts"] + df["home_pts"]
    df["goal_difference"] = df["home_pts"] - df["away_pts"]
    df["result"] = np.where(df["goal_difference"] > 0, 1, -1)

    model = BradleyTerry(df, None, 0.1)
    model.fit()

    home_team = "Richmond"
    away_team = "Geelong"
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=15, include_draw=False
    )

    # Make predictions
    # home_team = "Kolding"
    # away_team = "Bjerringbro/Silkeborg"
    # prob_home, prob_draw, prob_away = model.predict(home_team, away_team)
