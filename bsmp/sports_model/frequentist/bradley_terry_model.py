# %%

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from bsmp.sports_model.utils import dixon_coles_weights


class BradleyTerry:
    """
    Bradley-Terry model for predicting match outcomes.

    A probabilistic model that estimates team ratings and predicts match outcomes
    using maximum likelihood estimation. The model combines logistic regression for
    win probabilities with OLS regression for point spread predictions.

    Attributes:
        teams (np.ndarray): Unique team identifiers
        n_teams (int): Number of teams in the dataset
        team_map (Dict[str, int]): Mapping of team names to indices
        home_idx (np.ndarray): Indices of home teams
        away_idx (np.ndarray): Indices of away teams
        ratings_weights (np.ndarray): Weights for rating optimization
        match_weights (np.ndarray): Weights for spread prediction
        fitted (bool): Whether the model has been fitted
        params (np.ndarray): Optimized model parameters after fitting
            [0:n_teams] - Team ratings
            [-1] - Home advantage parameter
        intercept (float): Point spread model intercept
        spread_coefficient (float): Point spread model coefficient
        spread_error (float): Standard error of spread predictions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
        home_advantage: float = 0.1,
    ) -> None:
        """Initialize Bradley-Terry model."""
        # Extract numpy arrays once
        data = {
            col: df[col].to_numpy()
            for col in ["home_team", "away_team", "goal_difference", "result"]
        }
        self.__dict__.update(data)

        # Team setup
        self.teams = np.unique(self.home_team)
        self.n_teams = len(self.teams)
        self.team_map = {team: idx for idx, team in enumerate(self.teams)}

        # Create team indices
        self.home_idx = np.array([self.team_map[team] for team in self.home_team])
        self.away_idx = np.array([self.team_map[team] for team in self.away_team])

        # Set weights
        n_matches = len(df)
        self.ratings_weights = (
            np.ones(n_matches) if ratings_weights is None else ratings_weights
        )
        self.match_weights = (
            np.ones(n_matches) if match_weights is None else match_weights
        )

        # Initialize parameters
        self.params = np.zeros(self.n_teams + 1)
        self.params[-1] = home_advantage
        self.fitted = False

    def fit(self) -> None:
        """Fit the Bradley-Terry model."""
        try:
            # Optimize parameters
            self.params = self._optimize_parameters()

            # Fit point spread model
            rating_diff = self._get_rating_difference()
            (self.intercept, self.spread_coefficient), self.spread_error = (
                self._fit_ols(self.goal_difference, rating_diff)
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
        Predict match outcome probabilities or spread.

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

        # Calculate rating difference
        rating_diff = self._get_rating_difference(
            home_idx=self.team_map[home_team],
            away_idx=self.team_map[away_team],
        )

        # Calculate predicted spread
        predicted_spread = self.intercept + self.spread_coefficient * rating_diff

        if return_spread:
            return predicted_spread

        return self._calculate_probabilities(
            predicted_spread, self.spread_error, point_spread, include_draw
        )

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Calculate negative log likelihood for parameter optimization."""
        ratings = params[:-1]
        home_advantage = params[-1]
        log_likelihood = 0.0

        # Precompute home and away ratings
        home_ratings = ratings[self.home_idx]
        away_ratings = ratings[self.away_idx]
        win_probs = self._logit_transform(home_advantage + home_ratings - away_ratings)

        # Vectorized calculation
        win_mask = self.result == 1
        loss_mask = self.result == -1
        draw_mask = ~(win_mask | loss_mask)

        log_likelihood += np.sum(
            self.ratings_weights[win_mask] * np.log(win_probs[win_mask])
        )
        log_likelihood += np.sum(
            self.ratings_weights[loss_mask] * np.log(1 - win_probs[loss_mask])
        )
        log_likelihood += np.sum(
            self.ratings_weights[draw_mask]
            * (np.log(win_probs[draw_mask]) + np.log(1 - win_probs[draw_mask]))
        )

        return -log_likelihood

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP."""
        result = minimize(
            fun=lambda p: self._log_likelihood(p) / len(self.result),
            x0=self.params,
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
            home_idx, away_idx = self.home_idx, self.away_idx

        ratings = self.params[:-1]
        home_advantage = self.params[-1]
        return self._logit_transform(
            home_advantage + ratings[home_idx] - ratings[away_idx]
        )

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply logistic transformation."""
        return 1 / (1 + np.exp(-x))

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit weighted OLS regression using match weights."""
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights)

        # Use more efficient matrix operations
        XtW = X.T @ W
        coefficients = np.linalg.solve(XtW @ X, XtW @ y)
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
        """Calculate win/draw/loss probabilities using normal distribution."""
        if include_draw:
            thresholds = np.array([point_spread + 0.5, -point_spread - 0.5])
            probs = stats.norm.cdf(thresholds, predicted_spread, std_error)
            return 1 - probs[0], probs[0] - probs[1], probs[1]
        else:
            prob_home = 1 - stats.norm.cdf(point_spread, predicted_spread, std_error)
            return prob_home, np.nan, 1 - prob_home

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        return pd.DataFrame({"team": self.teams, "rating": self.params[:-1]}).set_index(
            "team"
        )

    def get_team_rating(self, team: str) -> float:
        """Get rating for a specific team."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        if team not in self.team_map:
            raise ValueError(f"Unknown team: {team}")

        return self.params[self.team_map[team]]


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
    away_team = "GWS Giants"

    # Use no weights
    model = BradleyTerry(df)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=15, include_draw=False
    )
