# %%
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from bsmp.sports_model.utils import dixon_coles_weights


class ZSD:
    """
    Z-Score Deviation (ZSD) model.

    A probabilistic model that predicts match outcomes using team-specific ratings and
    z-score transformations of expected scoring probabilities. The model uses weighted
    optimization to estimate team performance parameters and calculates win/draw/loss
    probabilities using a normal distribution.

    Attributes:
        teams (np.ndarray): Unique team identifiers
        team_map (dict): Mapping of team names to indices
        home_team_ratings (dict): Home performance ratings for each team
        away_team_ratings (dict): Away performance ratings for each team
        home_adj_factor (float): Home team adjustment factor
        away_adj_factor (float): Away team adjustment factor
        intercept (float): Model intercept term
        spread_coefficient (float): Coefficient for spread prediction
        spread_error (float): Standard error of spread predictions
        fitted (bool): Whether the model has been fitted
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Union[np.ndarray, None] = None,
        match_weights: Union[np.ndarray, None] = None,
    ):
        """
        Initialize ZSD model.

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
        self.away_team = df.away_team.to_numpy()
        self.home_pts = df.home_pts.to_numpy()
        self.away_pts = df.away_pts.to_numpy()
        self.goal_difference = df.goal_difference.to_numpy()

        # Set weights
        n_matches = len(df)
        self.ratings_weights = (
            np.ones(n_matches) if ratings_weights is None else ratings_weights
        )
        self.match_weights = (
            np.ones(n_matches) if match_weights is None else match_weights
        )

        self.team_map = {team: idx for idx, team in enumerate(self.teams)}

        # Calculate scoring statistics
        self._calculate_scoring_statistics()
        self._initialize_parameters()
        self.fitted = False

    def fit(self):
        """Fit the ZSD model to the training data."""
        # Optimize parameters
        self.params = self._optimize_parameters()
        self._update_team_ratings()

        # Calculate predicted points
        pred_scores = self._calculate_predicted_scores(
            self.data[:, 0].astype(int), self.data[:, 1].astype(int)
        )

        # Fit spread model
        raw_mov = pred_scores["home"] - pred_scores["away"]
        (self.intercept, self.spread_coefficient), self.spread_error = self._fit_ols(
            self.goal_difference, raw_mov
        )

        self.fitted = True

    def predict(
        self,
        home_team: str,
        away_team: str,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> Tuple[float, float, float]:
        """Predict match outcome probabilities."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        # Calculate predicted scores
        pred_scores = self._calculate_predicted_scores_for_teams(home_team, away_team)
        predicted_spread = self.intercept + self.spread_coefficient * (
            pred_scores["home"] - pred_scores["away"]
        )
        final_spread = self.intercept + self.spread_coefficient * predicted_spread

        return self._calculate_probabilities(
            final_spread, self.spread_error, point_spread, include_draw
        )

    def _calculate_scoring_statistics(self) -> None:
        """Calculate and store scoring statistics."""
        self.mean_home_score = np.mean(self.home_pts)
        self.mean_away_score = np.mean(self.away_pts)
        self.std_home_score = np.std(self.home_pts, ddof=1)
        self.std_away_score = np.std(self.away_pts, ddof=1)

        # Prepare match data array
        self.data = np.column_stack(
            [
                np.array([self.team_map[team] for team in self.home_team]),
                np.array([self.team_map[team] for team in self.away_team]),
                self.home_pts,
                self.away_pts,
            ]
        )

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        n_teams = len(self.teams)
        self.home_team_ratings = dict.fromkeys(self.teams, 1.0)
        self.away_team_ratings = dict.fromkeys(self.teams, 1.0)
        self.params = np.concatenate(
            [
                np.ones(n_teams),  # home ratings
                np.ones(n_teams),  # away ratings
                np.array([0.0, 0.0]),  # adjustment factors
            ]
        )

    def _calculate_predicted_scores(self, home_idx, away_idx) -> dict:
        """Calculate predicted scores for given team indices."""
        home_home, away_away, home_away, away_home = self._get_ratings(
            self.params[: len(self.teams)],
            self.params[len(self.teams) : 2 * len(self.teams)],
            home_idx,
            away_idx,
        )

        scores = {}
        for key, adj, h_rat, a_rat in [
            ("home", self.home_adj_factor, home_home, away_away),
            ("away", self.away_adj_factor, away_home, home_away),
        ]:
            param = self._parameter_estimate(adj, h_rat, a_rat)
            exp_prob = self._logit_transform(param)
            z_score = self._z_inverse(exp_prob)
            scores[key] = self._points_prediction(
                getattr(self, f"mean_{key}_score"),
                getattr(self, f"std_{key}_score"),
                z_score,
            )
        return scores

    def _calculate_predicted_scores_for_teams(
        self, home_team: str, away_team: str
    ) -> dict:
        """Calculate predicted scores for specific teams."""
        param_h = self._parameter_estimate(
            self.home_adj_factor,
            self.home_team_ratings[home_team],
            self.home_team_ratings[away_team],
        )
        param_a = self._parameter_estimate(
            self.away_adj_factor,
            self.away_team_ratings[home_team],
            self.away_team_ratings[away_team],
        )

        scores = {}
        for key, param in [("home", param_h), ("away", param_a)]:
            exp_prob = self._logit_transform(param)
            z_score = self._z_inverse(exp_prob)
            scores[key] = self._points_prediction(
                getattr(self, f"mean_{key}_score"),
                getattr(self, f"std_{key}_score"),
                z_score,
            )
        return scores

    def _update_team_ratings(self) -> None:
        """Update team ratings from optimized parameters."""
        n_teams = len(self.teams)
        self.home_team_ratings = dict(zip(self.teams, self.params[:n_teams]))
        self.away_team_ratings = dict(
            zip(self.teams, self.params[n_teams : 2 * n_teams])
        )
        self.home_adj_factor, self.away_adj_factor = self.params[-2:]

    def _optimize_parameters(self) -> np.ndarray:
        """
        Optimize model parameters using the SLSQP algorithm.

        Returns:
            np.ndarray: Optimized parameters [home_ratings, away_ratings, adj_factors]
        """
        # Constraint: mean team rating = 0 for identifiability
        n_teams = len(self.teams)
        constraints = [
            {"type": "eq", "fun": lambda p: np.mean(p[:n_teams])},
            {
                "type": "eq",
                "fun": lambda p: np.mean(p[n_teams : 2 * n_teams]),
            },
        ]

        # Add bounds to prevent numerical overflow
        bounds = [(-5, 5)] * (2 * n_teams)  # bounds for team ratings
        bounds.extend([(-2, 2)] * 2)  # bounds for adjustment factors

        result = minimize(
            fun=lambda p: self._sse_function(p, self.data, self.ratings_weights),
            x0=self.params,
            method="SLSQP",
            bounds=bounds,
            # constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 200},
        )

        return result.x

    def _sse_function(
        self, params: np.ndarray, data: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate the weighted sum of squared errors for given parameters."""
        n_teams = len(self.teams)
        home_ratings, away_ratings = params[:n_teams], params[n_teams : 2 * n_teams]
        home_adj_factor, away_adj_factor = params[-2:]

        # Get match data
        home_idx, away_idx = data[:, 0].astype(int), data[:, 1].astype(int)
        actual_scores = {"home": data[:, 2], "away": data[:, 3]}

        # Get team ratings and calculate predictions
        ratings = self._get_ratings(home_ratings, away_ratings, home_idx, away_idx)
        pred_scores = {}

        for key, (adj, h_rat, a_rat) in [
            ("home", (home_adj_factor, ratings[0], ratings[1])),
            ("away", (away_adj_factor, ratings[2], ratings[3])),
        ]:
            param = self._parameter_estimate(adj, h_rat, a_rat)
            z_score = self._z_inverse(self._logit_transform(param))
            pred_scores[key] = self._points_prediction(
                getattr(self, f"mean_{key}_score"),
                getattr(self, f"std_{key}_score"),
                z_score,
            )

        # Calculate weighted SSE
        errors = sum((actual_scores[k] - pred_scores[k]) ** 2 for k in ["home", "away"])
        return np.sum(weights * errors)

    def _get_ratings(self, home_ratings, away_ratings, home_team_idx, away_team_idx):
        home_home = home_ratings[home_team_idx]
        away_away = away_ratings[away_team_idx]
        home_away = home_ratings[away_team_idx]
        away_home = away_ratings[home_team_idx]
        return home_home, away_away, home_away, away_home

    def _points_prediction(self, mean_score, std_score, z_score):
        return mean_score + std_score * z_score

    def _parameter_estimate(self, adj_factor, home_ratings, away_ratings):
        return adj_factor + home_ratings - away_ratings

    def _z_inverse(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return stats.norm.ppf(z)

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply logistic transformation."""
        return 1 / (1 + np.exp(-x))

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit weighted OLS regression using match weights.

        Args:
            y (np.ndarray): Target variable (goal differences)
            X (np.ndarray): Feature matrix of team statistics

        Returns:
            Tuple[np.ndarray, float]: Model coefficients and standard error
        """
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights)  # Using match_weights for OLS

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
            prob_away = stats.norm.cdf(-point_spread, predicted_spread, std_error)
            prob_draw = np.nan

        return prob_home, prob_draw, prob_away


if __name__ == "__main__":
    # Load AFL data for testing
    df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv")  # .loc[:176]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["game_total"] = df["away_pts"] + df["home_pts"]
    df["goal_difference"] = df["home_pts"] - df["away_pts"]
    df["result"] = np.where(df["goal_difference"] > 0, 1, -1)

    df["date"] = pd.to_datetime(df["date"], format="%b %d (%a %I:%M%p)")
    team_weights = dixon_coles_weights(df.date, xi=0.08)
    np.random.seed(0)
    spread_weights = np.random.uniform(0.1, 1.0, len(df))

    home_team = "St Kilda"
    away_team = "North Melbourne"

    # Use no weights
    model = ZSD(df)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )

    model = ZSD(df, ratings_weights=team_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )
    # Use different weights
    model = ZSD(df, ratings_weights=team_weights, match_weights=spread_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=0, include_draw=False
    )

# %%
