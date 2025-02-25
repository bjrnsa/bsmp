# %%
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.utils import dixon_coles_weights

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


class ZSD:
    """
    Z-Score Deviation (ZSD) model for predicting sports match outcomes with scikit-learn-like API.

    The model uses weighted optimization to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

    Parameters
    ----------
    None

    Attributes
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    n_teams_ : int
        Number of teams in the dataset
    team_map_ : Dict[str, int]
        Mapping of team names to indices
    home_idx_ : np.ndarray
        Indices of home teams
    away_idx_ : np.ndarray
        Indices of away teams
    ratings_weights_ : np.ndarray
        Weights for rating optimization
    match_weights_ : np.ndarray
        Weights for spread prediction
    is_fitted_ : bool
        Whether the model has been fitted
    params_ : np.ndarray
        Optimized model parameters after fitting
        [0:n_teams_] - Offensive ratings
        [n_teams_:2*n_teams_] - Defensive ratings
        [-2:] - Home/away adjustment factors
    mean_home_score_ : float
        Mean home team score
    std_home_score_ : float
        Standard deviation of home team scores
    mean_away_score_ : float
        Mean away team score
    std_away_score_ : float
        Standard deviation of away team scores
    intercept_ : float
        Spread model intercept
    spread_coefficient_ : float
        Spread model coefficient
    spread_error_ : float
        Standard error of spread predictions

    Note
    ----
    The model ensures that both offensive and defensive ratings sum to zero
    through optimization constraints, making the ratings interpretable as
    relative performance measures.
    """

    NAME = "ZSD"

    def __init__(self) -> None:
        """Initialize ZSD model."""
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: pd.DataFrame = None,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
        initial_params: Optional[Union[np.ndarray, dict]] = None,
    ) -> "ZSD":
        """
        Fit the ZSD model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with at least two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
            If y is None, X must have a third column with goal differences.
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Goal differences (home - away). If provided, this will be used instead of
            the third column in X.
        Z : pd.DataFrame
            Additional data for the model, containing at least 'home_goals' and 'away_goals' columns.
        ratings_weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        match_weights : Optional[np.ndarray], default=None
            Weights for spread prediction
        initial_params : Optional[Union[np.ndarray, dict]], default=None
            Optional initial parameter values

        Returns
        -------
        self : ZSD
            The fitted ZSD model instance.
        """
        try:
            # Validate input dimensions and types
            self._validate_X(X)

            # Validate Z is provided and has required dimensions
            if Z is None:
                raise ValueError(
                    "Z must be provided with home_goals and away_goals data"
                )
            if len(Z) != len(X):
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

            self.home_goals_ = Z.iloc[:, 0].to_numpy()
            self.away_goals_ = Z.iloc[:, 1].to_numpy()

            if len(self.home_goals_) == 0 or len(self.away_goals_) == 0:
                raise ValueError("Empty input data")

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

            self._calculate_scoring_statistics()

            # Optimize
            self.params_ = self._optimize_parameters(initial_params)

            # Fit spread model
            pred_scores = self._predict_scores()
            raw_mov = pred_scores["home"] - pred_scores["away"]
            (self.intercept_, self.spread_coefficient_), self.spread_error_ = (
                self._fit_ols(self.goal_difference_, raw_mov)
            )

            self.is_fitted_ = True
            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def predict(
        self,
        X: pd.DataFrame,
        Z: pd.DataFrame = None,
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
        Z : pd.DataFrame
            Additional data for prediction. Not used in this method but included for API consistency.
        point_spread : float, default=0.0
            Point spread adjustment

        Returns
        -------
        np.ndarray
            Predicted point spreads (goal differences)
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Get predicted scores using team indices
            pred_scores = self._predict_scores(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )

            # Calculate spread
            raw_mov = pred_scores["home"] - pred_scores["away"]
            predicted_spreads[i] = self.intercept_ + self.spread_coefficient_ * raw_mov

        return predicted_spreads

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: pd.DataFrame = None,
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
        Z : pd.DataFrame
            Additional data for prediction. Not used in this method but included for API consistency.
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

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        n_classes = 3 if include_draw else 2
        probabilities = np.zeros((len(X), n_classes))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Get predicted scores using team indices
            pred_scores = self._predict_scores(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )

            # Calculate spread
            raw_mov = pred_scores["home"] - pred_scores["away"]
            predicted_spread = self.intercept_ + self.spread_coefficient_ * raw_mov

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

    def _optimize_parameters(
        self, initial_params: Optional[Union[np.ndarray, dict]] = None
    ) -> np.ndarray:
        """
        Optimize model parameters using SLSQP optimization.

        Parameters
        ----------
        initial_params : Optional[Union[np.ndarray, dict]], default=None
            Optional initial parameter values

        Returns
        -------
        np.ndarray
            Optimized parameters

        Raises
        ------
        RuntimeError
            If optimization fails
        """
        constraints = [
            {"type": "eq", "fun": lambda p: np.mean(p[: self.n_teams_])},
            {
                "type": "eq",
                "fun": lambda p: np.mean(p[self.n_teams_ : 2 * self.n_teams_]),
            },
        ]

        bounds = [(-50, 50)] * (2 * self.n_teams_) + [(-np.inf, np.inf)] * 2
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
        """
        Calculate the weighted sum of squared errors for given parameters.

        Parameters
        ----------
        params : np.ndarray
            Model parameters

        Returns
        -------
        float
            Weighted sum of squared errors
        """
        # Unpack parameters efficiently
        pred_scores = self._predict_scores(
            self.home_idx_,
            self.away_idx_,
            *np.split(params, [self.n_teams_, 2 * self.n_teams_]),
        )
        squared_errors = (self.home_goals_ - pred_scores["home"]) ** 2 + (
            self.away_goals_ - pred_scores["away"]
        ) ** 2
        return np.sum(self.ratings_weights_ * squared_errors)

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

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None], default=None
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None], default=None
            Index(es) of away team(s)
        home_ratings : Union[np.ndarray, None], default=None
            Optional home ratings to use
        away_ratings : Union[np.ndarray, None], default=None
            Optional away ratings to use
        factors : Union[Tuple[float, float], None], default=None
            Optional (home_factor, away_factor) tuple

        Returns
        -------
        Dict[str, np.ndarray]
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params_[-2:]

        ratings = self._get_team_ratings(home_idx, away_idx, home_ratings, away_ratings)

        return {
            "home": self._transform_to_score(
                self._parameter_estimate(
                    factors[0], ratings["home_rating"], ratings["away_rating"]
                ),
                self.mean_home_score_,
                self.std_home_score_,
            ),
            "away": self._transform_to_score(
                self._parameter_estimate(
                    factors[1], ratings["home_away_rating"], ratings["away_home_rating"]
                ),
                self.mean_away_score_,
                self.std_away_score_,
            ),
        }

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None],
        away_idx: Union[int, np.ndarray, None],
        home_ratings: Union[np.ndarray, None] = None,
        away_ratings: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract team ratings from parameters.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None]
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None]
            Index(es) of away team(s)
        home_ratings : Union[np.ndarray, None], default=None
            Optional home ratings to use
        away_ratings : Union[np.ndarray, None], default=None
            Optional away ratings to use

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with team ratings
        """
        if home_ratings is None:
            home_ratings, away_ratings = np.split(self.params_[: 2 * self.n_teams_], 2)
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

        return {
            "home_rating": home_ratings[home_idx],
            "home_away_rating": away_ratings[home_idx],
            "away_rating": home_ratings[away_idx],
            "away_home_rating": away_ratings[away_idx],
        }

    def _parameter_estimate(
        self, adj_factor: float, home_rating: np.ndarray, away_rating: np.ndarray
    ) -> np.ndarray:
        """
        Calculate parameter estimate for score prediction.

        Parameters
        ----------
        adj_factor : float
            Adjustment factor
        home_rating : np.ndarray
            Home team rating
        away_rating : np.ndarray
            Away team rating

        Returns
        -------
        np.ndarray
            Parameter estimate
        """
        return adj_factor + home_rating - away_rating

    def _transform_to_score(
        self, param: np.ndarray, mean: float, std: float
    ) -> np.ndarray:
        """
        Transform parameter to actual score prediction.

        Parameters
        ----------
        param : np.ndarray
            Parameter value
        mean : float
            Mean score
        std : float
            Standard deviation of scores

        Returns
        -------
        np.ndarray
            Predicted score
        """
        exp_prob = self._logit_transform(param)
        z_score = self._z_inverse(exp_prob)
        return mean + std * z_score

    def _z_inverse(self, prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate inverse of standard normal CDF.

        Parameters
        ----------
        prob : Union[float, np.ndarray]
            Probability value(s)

        Returns
        -------
        Union[float, np.ndarray]
            Z-score(s)
        """
        return stats.norm.ppf(prob)

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply logistic transformation with numerical stability.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Input value(s)

        Returns
        -------
        Union[float, np.ndarray]
            Transformed value(s)
        """
        # Clip values to avoid overflow
        x_clipped = np.clip(x, -700, 700)  # exp(700) is close to float max
        return 1 / (1 + np.exp(-x_clipped))

    def _calculate_scoring_statistics(self) -> None:
        """Calculate and store scoring statistics for home and away teams."""
        # Calculate all statistics in one pass using numpy
        home_stats: np.ndarray = np.array(
            [np.mean(self.home_goals_), np.std(self.home_goals_, ddof=1)]
        )
        away_stats: np.ndarray = np.array(
            [np.mean(self.away_goals_), np.std(self.away_goals_, ddof=1)]
        )

        # Unpack results
        self.mean_home_score_: float = home_stats[0]
        self.std_home_score_: float = home_stats[1]
        self.mean_away_score_: float = away_stats[0]
        self.std_away_score_: float = away_stats[1]

        # Validate statistics
        if not (self.std_home_score_ > 0 and self.std_away_score_ > 0):
            raise ValueError(
                "Invalid scoring statistics: zero or negative standard deviation"
            )

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit weighted OLS regression using match weights.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable
        X : np.ndarray
            The independent variables

        Returns
        -------
        Tuple[np.ndarray, float]
            Coefficients and standard error
        """
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights_)

        # Use more efficient matrix operations
        XtW = X.T @ W
        coefficients = np.linalg.solve(XtW @ X, XtW @ y)
        residuals = y - X @ coefficients
        weighted_sse = residuals.T @ W @ residuals
        std_error = np.sqrt(weighted_sse / (len(y) - X.shape[1]))

        return coefficients, std_error

    def _get_initial_params(
        self, initial_params: Optional[Union[np.ndarray, dict]]
    ) -> np.ndarray:
        """
        Generate initial parameters, incorporating any provided values.

        Parameters
        ----------
        initial_params : Optional[Union[np.ndarray, dict]]
            Optional initial parameter values as either:
                - np.ndarray: Full parameter vector of length 2*n_teams_ + 2
                - dict: Parameters with keys:
                    'teams': list of team names in order
                    'home': list of home ratings
                    'away': list of away ratings
                    'factors': tuple of (home_factor, away_factor)
                - None: Use random initialization

        Returns
        -------
        np.ndarray
            Complete parameter vector

        Raises
        ------
        ValueError
            If parameters are invalid or don't match teams
        """
        if initial_params is None:
            return np.random.normal(0, 0.1, 2 * self.n_teams_ + 2)

        if isinstance(initial_params, np.ndarray):
            if len(initial_params) != 2 * self.n_teams_ + 2:
                raise ValueError(f"Expected {2 * self.n_teams_ + 2} parameters")
            return initial_params.copy()

        if not isinstance(initial_params, dict):
            raise ValueError("initial_params must be array, dict, or None")

        # Verify teams match
        if "teams" not in initial_params:
            raise ValueError("Missing 'teams' in initial_params")
        if set(initial_params["teams"]) != set(self.teams_):
            raise ValueError("Provided teams don't match model teams")

        # Create parameter vector
        x0 = np.zeros(2 * self.n_teams_ + 2)

        # Map parameters according to team order
        team_indices = {team: i for i, team in enumerate(initial_params["teams"])}
        model_indices = {i: team_indices[team] for i, team in enumerate(self.teams_)}

        # Set home ratings
        if "home" in initial_params:
            home = initial_params["home"]
            if len(home) != self.n_teams_:
                raise ValueError(f"Expected {self.n_teams_} home ratings")
            for i in range(self.n_teams_):
                x0[i] = home[model_indices[i]]

        # Set away ratings
        if "away" in initial_params:
            away = initial_params["away"]
            if len(away) != self.n_teams_:
                raise ValueError(f"Expected {self.n_teams_} away ratings")
            for i in range(self.n_teams_):
                x0[i + self.n_teams_] = away[model_indices[i]]

        # Set factors
        if "factors" in initial_params:
            factors = initial_params["factors"]
            if len(factors) != 2:
                raise ValueError("Expected 2 factors (home, away)")
            x0[-2:] = factors

        return x0

    def _validate_X(self, X: pd.DataFrame, fit: bool = True) -> None:
        """
        Validate input DataFrame dimensions and types.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        fit : bool, default=True
            Whether this is being called during fit (requires at least 2 columns)
            or during predict (requires exactly 2 columns)
        """
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Check minimum number of columns
        min_cols = 2
        if X.shape[1] < min_cols:
            raise ValueError(f"X must have at least {min_cols} columns")

        # For predict methods, exactly 2 columns are required
        if not fit and X.shape[1] != 2:
            raise ValueError("X must have exactly 2 columns for prediction")

        # Check that first two columns contain strings (team names)
        for i in range(2):
            if not pd.api.types.is_string_dtype(X.iloc[:, i]):
                raise ValueError(f"Column {i} must contain string values (team names)")

    def _validate_teams(self, teams: List[str]) -> None:
        """
        Validate teams exist in the model.

        Parameters
        ----------
        teams : List[str]
            List of team names to validate
        """
        for team in teams:
            if team not in self.team_map_:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Team ratings with columns ['team', 'home', 'away']
        """
        self._check_is_fitted()

        home_ratings = self.params_[: self.n_teams_]
        away_ratings = self.params_[self.n_teams_ : 2 * self.n_teams_]

        return pd.DataFrame(
            {"team": self.teams_, "home": home_ratings, "away": away_ratings}
        ).set_index("team")

    def get_team_rating(self, team: str) -> Tuple[float, float]:
        """
        Get ratings for a specific team.

        Parameters
        ----------
        team : str
            Name of the team

        Returns
        -------
        Tuple[float, float]
            Tuple of (home_rating, away_rating)
        """
        self._check_is_fitted()
        self._validate_teams([team])

        idx = self.team_map_[team]
        home_rating = self.params_[idx]
        away_rating = self.params_[idx + self.n_teams_]
        return home_rating, away_rating


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
    team_weights = dixon_coles_weights(train_df.datetime, xi=0.018)

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = ZSD()

    # Prepare training data
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    Z_train = train_df[["home_goals", "away_goals"]]
    model.fit(X_train, y_train, Z=Z_train, ratings_weights=team_weights)

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


# %%
