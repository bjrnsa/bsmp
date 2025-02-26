# %%
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.base_model import BaseModel
from bsmp.sports_model.utils import dixon_coles_weights


class Poisson(BaseModel):
    """
    Dixon and Coles adjusted Poisson model for predicting outcomes of football
    (soccer) matches with scikit-learn-like API.

    A probabilistic model that estimates team attack/defense strengths and predicts
    match outcomes using maximum likelihood estimation with a Poisson distribution
    and Dixon-Coles adjustment for low-scoring games.

    Parameters
    ----------
    home_advantage : float, default=0.25
        Initial value for home advantage parameter

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
        [0:n_teams_] - Team attack ratings
        [n_teams_:2*n_teams_] - Team defense ratings
        [-2] - Home advantage parameter
        [-1] - Rho parameter
    """

    NAME = "DC"

    def __init__(self, home_advantage: float = 0.25) -> None:
        """Initialize Dixon-Coles model."""
        super().__init__()
        self.home_advantage_ = home_advantage
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "Poisson":
        """
        Fit the Dixon-Coles model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Not used in this model but included for API consistency
        Z : pd.DataFrame
            Additional data for the model with home_goals and away_goals
            Must be provided with exactly 2 columns
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization

        Returns
        -------
        self : DixonColes
            Fitted model
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
            if Z.shape[1] != 2:
                raise ValueError(
                    "Z must have exactly 2 columns: home_goals and away_goals"
                )

            # Extract team data
            self.home_team_ = X.iloc[:, 0].to_numpy()
            self.away_team_ = X.iloc[:, 1].to_numpy()
            self.home_goals_ = Z.iloc[:, 0].to_numpy()
            self.away_goals_ = Z.iloc[:, 1].to_numpy()

            # Team setup
            self.teams_ = np.sort(
                np.unique(np.concatenate([self.home_team_, self.away_team_]))
            )
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
            self.weights_ = np.ones(n_matches) if weights is None else weights

            # Initialize parameters
            self.params_ = np.concatenate(
                (
                    [1] * self.n_teams_,  # attack
                    [-1] * self.n_teams_,  # defense
                    [self.home_advantage_],  # home advantage
                )
            )

            # Optimize parameters
            options = {"maxiter": 100, "disp": False}
            constraints = [
                {"type": "eq", "fun": lambda x: sum(x[: self.n_teams_]) - self.n_teams_}
            ]
            bounds = [(-3, 3)] * self.n_teams_  # attack
            bounds += [(-3, 3)] * self.n_teams_  # defense
            bounds += [(0, 2)]  # home advantage

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._res = minimize(
                    self._fit,
                    self.params_,
                    constraints=constraints,
                    bounds=bounds,
                    options=options,
                )

            self.params_ = self._res["x"]
            self.n_params_ = len(self.params_)

            home_teams = X.iloc[:, 0].to_numpy()
            away_teams = X.iloc[:, 1].to_numpy()
            predicted_spreads = np.zeros(len(X))

            for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
                # Validate teams
                self._validate_teams([home_team, away_team])

                # Get team indices
                home_idx = self.team_map_[home_team]
                away_idx = self.team_map_[away_team]

                # Get parameters
                home_attack = self.params_[home_idx]
                away_attack = self.params_[away_idx]
                home_defence = self.params_[home_idx + self.n_teams_]
                away_defence = self.params_[away_idx + self.n_teams_]
                home_advantage = self.params_[-2]

                # Calculate expected goals
                home_goals = np.exp(home_advantage + home_attack + away_defence)
                away_goals = np.exp(away_attack + home_defence)
                predicted_spreads[i] = home_goals - away_goals

            # Calculate spread error
            residuals = y - predicted_spreads
            sse = np.sum((residuals**2))
            self.spread_error_ = np.sqrt(sse / (len(X) - X.shape[1]))

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
            Not used in this model but included for API consistency
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

            # Get team indices
            home_idx = self.team_map_[home_team]
            away_idx = self.team_map_[away_team]

            # Get parameters
            home_attack = self.params_[home_idx]
            away_attack = self.params_[away_idx]
            home_defence = self.params_[home_idx + self.n_teams_]
            away_defence = self.params_[away_idx + self.n_teams_]
            home_advantage = self.params_[-1]

            # Calculate expected goals
            home_goals = np.exp(home_advantage + home_attack + away_defence)
            away_goals = np.exp(away_attack + home_defence)
            predicted_spreads[i] = home_goals - away_goals

        return predicted_spreads

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
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
            Not used in this model but included for API consistency
        point_spread : float, default=0.0
            Point spread adjustment
        include_draw : bool, default=True
            Whether to include draw probability
        outcome: Optional[str], default=None
            Outcome to predict (home_win, draw, away_win)

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

        if outcome is None:
            n_classes = 3 if include_draw else 2
            probabilities = np.zeros((len(X), n_classes))
        else:
            probabilities = np.zeros((len(X),))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Get team indices
            home_idx = self.team_map_[home_team]
            away_idx = self.team_map_[away_team]

            # Get parameters
            home_attack = self.params_[home_idx]
            away_attack = self.params_[away_idx]
            home_defence = self.params_[home_idx + self.n_teams_]
            away_defence = self.params_[away_idx + self.n_teams_]
            home_advantage = self.params_[-1]

            # Calculate expected goals
            home_goals = np.exp(home_advantage + home_attack + away_defence)
            away_goals = np.exp(away_attack + home_defence)

            # Calculate spread
            predicted_spread = home_goals - away_goals

            # Calculate probabilities
            if include_draw:
                thresholds = np.array([point_spread + 0.5, -point_spread - 0.5])
                probs = norm.cdf(thresholds, predicted_spread, self.spread_error_)
                prob_home, prob_draw, prob_away = (
                    1 - probs[0],
                    probs[0] - probs[1],
                    probs[1],
                )
            else:
                prob_home = 1 - norm.cdf(
                    point_spread, predicted_spread, self.spread_error_
                )
                prob_home, prob_away = prob_home, 1 - prob_home

            if outcome is not None:
                if outcome == "home_win":
                    probabilities[i] = prob_home
                elif outcome == "away_win":
                    probabilities[i] = prob_away
                elif outcome == "draw":
                    probabilities[i] = prob_draw
            else:
                if include_draw:
                    probabilities[i] = [prob_home, prob_draw, prob_away]
                else:
                    probabilities[i] = [prob_home, prob_away]

        if outcome:
            return probabilities.reshape(-1)
        return probabilities

    def get_params(self) -> dict:
        """
        Get the current parameters of the model.

        Returns
        -------
        dict
            Dictionary containing model parameters
        """
        self._check_is_fitted()
        return {
            "teams": self.teams_,
            "team_map": self.team_map_,
            "params": self.params_,
            "is_fitted": self.is_fitted_,
            "home_advantage": self.home_advantage_,
        }

    def set_params(self, params: dict) -> None:
        """
        Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.teams_ = params["teams"]
        self.team_map_ = params["team_map"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]
        self.home_advantage_ = params["home_advantage"]

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with team ratings
        """
        self._check_is_fitted()
        ratings_df = pd.DataFrame(
            {
                "attack": self.params_[: self.n_teams_],
                "defense": self.params_[self.n_teams_ : 2 * self.n_teams_],
            },
            index=self.teams_,
        )
        ratings_df.loc["Home Advantage"] = [self.params_[-1], np.nan]

        return ratings_df.round(3)

    def _fit(self, params: np.ndarray) -> float:
        """
        Calculate negative log likelihood for parameter optimization.

        Parameters
        ----------
        params : np.ndarray
            Model parameters to optimize
        fixtures : pd.DataFrame
            Match data
        teams : np.ndarray
            Unique team names

        Returns
        -------
        float
            Negative log likelihood
        """
        # Extract parameters
        attack_ratings = params[: self.n_teams_]
        defense_ratings = params[self.n_teams_ : 2 * self.n_teams_]
        home_advantage = params[-1]

        # Get team ratings for home and away teams
        home_attack = attack_ratings[self.home_idx_]
        away_attack = attack_ratings[self.away_idx_]
        home_defense = defense_ratings[self.home_idx_]
        away_defense = defense_ratings[self.away_idx_]

        # Calculate expected goals
        home_exp = np.exp(home_advantage + home_attack + away_defense)
        away_exp = np.exp(away_attack + home_defense)

        # Calculate log probabilities
        home_llk = poisson.logpmf(self.home_goals_, home_exp)
        away_llk = poisson.logpmf(self.away_goals_, away_exp)

        # Sum log likelihood with weights
        log_likelihood = np.sum(self.weights_ * (home_llk + away_llk))

        return -log_likelihood


# %%
if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        # seasons=["2024/2025"],
    )
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
    weights = dixon_coles_weights(train_df.datetime, xi=0.0018)

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = Poisson()

    # Prepare training data
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    Z_train = train_df[["home_goals", "away_goals"]]
    model.fit(X_train, y_train, Z=Z_train, weights=weights)

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
