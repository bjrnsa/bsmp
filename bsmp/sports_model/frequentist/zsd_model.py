# %%
import warnings
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.utils import dixon_coles_weights

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


class ZSD:
    """
    Z-Score Deviation (ZSD) model for predicting sports match outcomes.

    The model uses weighted optimization to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

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
            [0:n_teams] - Offensive ratings
            [n_teams:2*n_teams] - Defensive ratings
            [-2:] - Home/away adjustment factors
        mean_home_score (float): Mean home team score
        std_home_score (float): Standard deviation of home team scores
        mean_away_score (float): Mean away team score
        std_away_score (float): Standard deviation of away team scores
        intercept (float): Spread model intercept
        spread_coefficient (float): Spread model coefficient
        spread_error (float): Standard error of spread predictions

    Note:
        The model ensures that both offensive and defensive ratings sum to zero
        through optimization constraints, making the ratings interpretable as
        relative performance measures.
    """

    NAME = "ZSD"

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Union[np.ndarray, None] = None,
        match_weights: Union[np.ndarray, None] = None,
    ) -> None:
        """Initialize ZSD model."""
        # Extract numpy arrays once
        data = {
            col: df[col].to_numpy()
            for col in [
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "goal_difference",
            ]
        }
        self.__dict__.update(data)  # Add arrays directly to instance

        # Team setup
        self.teams: np.ndarray = np.unique(self.home_team)
        self.n_teams: int = len(self.teams)
        self.team_map: Dict[str, int] = {
            team: idx for idx, team in enumerate(self.teams)
        }

        # Create team indices
        self.home_idx: np.ndarray = np.array(
            [self.team_map[team] for team in self.home_team]
        )
        self.away_idx: np.ndarray = np.array(
            [self.team_map[team] for team in self.away_team]
        )

        # Set weights
        n_matches: int = len(df)
        self.ratings_weights: np.ndarray = (
            np.ones(n_matches) if ratings_weights is None else ratings_weights
        )
        self.match_weights: np.ndarray = (
            np.ones(n_matches) if match_weights is None else match_weights
        )

        self.fitted: bool = False

    def fit(self, initial_params: Union[np.ndarray, dict, None] = None) -> "ZSD":
        """Fit the ZSD model to the training data."""
        try:
            if len(self.home_goals) == 0 or len(self.away_goals) == 0:
                raise ValueError("Empty input data")

            self._calculate_scoring_statistics()

            # Optimize
            self.params = self._optimize_parameters(initial_params)

            # Fit spread model
            pred_scores = self._predict_scores()
            raw_mov = pred_scores["home"] - pred_scores["away"]
            (self.intercept, self.spread_coefficient), self.spread_error = (
                self._fit_ols(self.goal_difference, raw_mov)
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
            home_team: Name of the home team
            away_team: Name of the away team
            point_spread: Points handicap (default: 0.0)
            include_draw: Whether to include draw probability (default: True)
            return_spread: If True, return predicted spread instead of probabilities (default: False)

        Returns:
            If return_spread=False: Tuple[float, float, float] (home_win_prob, draw_prob, away_win_prob)
            If return_spread=True: float (predicted spread)
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        # Validate teams
        for team in (home_team, away_team):
            if team not in self.team_map:
                raise ValueError(f"Unknown team: {team}")

        # Get predicted scores using team indices
        pred_scores = self._predict_scores(
            home_idx=self.team_map[home_team], away_idx=self.team_map[away_team]
        )

        # Calculate spread
        raw_mov = pred_scores["home"] - pred_scores["away"]
        predicted_spread = self.intercept + self.spread_coefficient * raw_mov

        if return_spread:
            return predicted_spread

        return self._calculate_probabilities(
            predicted_spread, self.spread_error, point_spread, include_draw
        )

    def _optimize_parameters(
        self, initial_params: Union[np.ndarray, dict, None] = None
    ) -> np.ndarray:
        """
        Optimize model parameters using SLSQP optimization.

        Args:
            initial_params: Optional initial parameter values

        Returns:
            np.ndarray: Optimized parameters

        Raises:
            RuntimeError: If optimization fails
        """
        constraints = [
            {"type": "eq", "fun": lambda p: np.mean(p[: self.n_teams])},
            {
                "type": "eq",
                "fun": lambda p: np.mean(p[self.n_teams : 2 * self.n_teams]),
            },
        ]

        bounds = [(-50, 50)] * (2 * self.n_teams) + [(-np.inf, np.inf)] * 2
        x0 = self._get_initial_params(initial_params)

        result = minimize(
            fun=self._sse_function,
            x0=x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 100000, "ftol": 1e-8},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x

    def _sse_function(self, params) -> float:
        """Calculate the weighted sum of squared errors for given parameters."""
        # Unpack parameters efficiently
        pred_scores = self._predict_scores(
            self.home_idx,
            self.away_idx,
            *np.split(params, [self.n_teams, 2 * self.n_teams]),
        )
        squared_errors = (self.home_goals - pred_scores["home"]) ** 2 + (
            self.away_goals - pred_scores["away"]
        ) ** 2
        return np.sum(self.ratings_weights * squared_errors)

    def _predict_scores(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
        home_ratings: Union[np.ndarray, None] = None,
        away_ratings: Union[np.ndarray, None] = None,
        factors: Union[Tuple[float, float], None] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate predicted scores using team ratings and factors.

        Args:
            home_idx: Index(es) of home team(s)
            away_idx: Index(es) of away team(s)
            home_ratings: Optional home ratings to use
            away_ratings: Optional away ratings to use
            factors: Optional (home_factor, away_factor) tuple

        Returns:
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params[-2:]

        ratings = self._get_team_ratings(home_idx, away_idx, home_ratings, away_ratings)

        return {
            "home": self._transform_to_score(
                self._parameter_estimate(
                    factors[0], ratings["home_offense"], ratings["away_offense"]
                ),
                self.mean_home_score,
                self.std_home_score,
            ),
            "away": self._transform_to_score(
                self._parameter_estimate(
                    factors[1], ratings["home_defense"], ratings["away_defense"]
                ),
                self.mean_away_score,
                self.std_away_score,
            ),
        }

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None],
        away_idx: Union[int, np.ndarray, None],
        home_ratings: Union[np.ndarray, None] = None,
        away_ratings: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract team ratings from parameters."""
        if home_ratings is None:
            home_ratings, away_ratings = np.split(self.params[: 2 * self.n_teams], 2)
        if home_idx is None:
            home_idx, away_idx = self.home_idx, self.away_idx

        return {
            "home_offense": home_ratings[home_idx],
            "home_defense": away_ratings[home_idx],
            "away_offense": home_ratings[away_idx],
            "away_defense": away_ratings[away_idx],
        }

    def _parameter_estimate(
        self, adj_factor: float, home_rating: np.ndarray, away_rating: np.ndarray
    ) -> np.ndarray:
        """Calculate parameter estimate for score prediction."""
        return adj_factor + home_rating - away_rating

    def _transform_to_score(
        self, param: np.ndarray, mean: float, std: float
    ) -> np.ndarray:
        """Transform parameter to actual score prediction."""
        exp_prob = self._logit_transform(param)
        z_score = self._z_inverse(exp_prob)
        return mean + std * z_score

    def _z_inverse(self, prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate inverse of standard normal CDF."""
        return stats.norm.ppf(prob)

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply logistic transformation with numerical stability."""
        # Clip values to avoid overflow
        x_clipped = np.clip(x, -700, 700)  # exp(700) is close to float max
        return 1 / (1 + np.exp(-x_clipped))

    def _points_prediction(self, mean_score, std_score, z_score):
        return mean_score + std_score * z_score

    def _calculate_scoring_statistics(self) -> None:
        """Calculate and store scoring statistics for home and away teams."""
        # Calculate all statistics in one pass using numpy
        home_stats: np.ndarray = np.array(
            [np.mean(self.home_goals), np.std(self.home_goals, ddof=1)]
        )
        away_stats: np.ndarray = np.array(
            [np.mean(self.away_goals), np.std(self.away_goals, ddof=1)]
        )

        # Unpack results
        self.mean_home_score: float = home_stats[0]
        self.std_home_score: float = home_stats[1]
        self.mean_away_score: float = away_stats[0]
        self.std_away_score: float = away_stats[1]

        # Validate statistics
        if not (self.std_home_score > 0 and self.std_away_score > 0):
            raise ValueError(
                "Invalid scoring statistics: zero or negative standard deviation"
            )

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

    def _get_initial_params(
        self, initial_params: Union[np.ndarray, dict, None]
    ) -> np.ndarray:
        """
        Generate initial parameters, incorporating any provided values.

        Args:
            initial_params: Optional initial parameter values as either:
                - np.ndarray: Full parameter vector of length 2*n_teams + 2
                - dict: Parameters with keys:
                    'teams': list of team names in order
                    'home': list of home ratings
                    'away': list of away ratings
                    'factors': tuple of (home_factor, away_factor)
                - None: Use random initialization

        Returns:
            np.ndarray: Complete parameter vector

        Raises:
            ValueError: If parameters are invalid or don't match teams
        """
        if initial_params is None:
            return np.random.normal(0, 0.1, 2 * self.n_teams + 2)

        if isinstance(initial_params, np.ndarray):
            if len(initial_params) != 2 * self.n_teams + 2:
                raise ValueError(f"Expected {2 * self.n_teams + 2} parameters")
            return initial_params.copy()

        if not isinstance(initial_params, dict):
            raise ValueError("initial_params must be array, dict, or None")

        # Verify teams match
        if "teams" not in initial_params:
            raise ValueError("Missing 'teams' in initial_params")
        if set(initial_params["teams"]) != set(self.teams):
            raise ValueError("Provided teams don't match model teams")

        # Create parameter vector
        x0 = np.zeros(2 * self.n_teams + 2)

        # Map parameters according to team order
        team_indices = {team: i for i, team in enumerate(initial_params["teams"])}
        model_indices = {i: team_indices[team] for i, team in enumerate(self.teams)}

        # Set home ratings
        if "home" in initial_params:
            home = initial_params["home"]
            if len(home) != self.n_teams:
                raise ValueError(f"Expected {self.n_teams} home ratings")
            for i in range(self.n_teams):
                x0[i] = home[model_indices[i]]

        # Set away ratings
        if "away" in initial_params:
            away = initial_params["away"]
            if len(away) != self.n_teams:
                raise ValueError(f"Expected {self.n_teams} away ratings")
            for i in range(self.n_teams):
                x0[i + self.n_teams] = away[model_indices[i]]

        # Set factors
        if "factors" in initial_params:
            factors = initial_params["factors"]
            if len(factors) != 2:
                raise ValueError("Expected 2 factors (home, away)")
            x0[-2:] = factors

        return x0

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns:
            pd.DataFrame: Team ratings with columns ['team', 'home', 'away']

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        home_ratings = self.params[: self.n_teams]
        away_ratings = self.params[self.n_teams : 2 * self.n_teams]

        return pd.DataFrame(
            {"team": self.teams, "home": home_ratings, "away": away_ratings}
        ).set_index("team")

    def get_team_rating(self, team: str, home: bool = True) -> float:
        """
        Get ratings for a specific team.

        Args:
            team: Name of the team

        Returns:
            Dict with keys 'home' and 'away'

        Raises:
            ValueError: If team not found or model not fitted
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        if team not in self.team_map:
            raise ValueError(f"Unknown team: {team}")

        idx = self.team_map[team]
        home_rating = self.params[idx]
        away_rating = self.params[idx]
        if home:
            return home_rating
        else:
            return away_rating


if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )
    team_weights = dixon_coles_weights(df.datetime)

    home_team = "GOG"
    away_team = "Mors"

    model = ZSD(df, ratings_weights=team_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=2, include_draw=True
    )
    print(f"Home win probability: {prob_home}")
    print(f"Draw probability: {prob_draw}")
    print(f"Away win probability: {prob_away}")
    print(f"Home rating: {model.get_team_rating(home_team)}")
    print(f"Away rating: {model.get_team_rating(away_team)}")


# %%
