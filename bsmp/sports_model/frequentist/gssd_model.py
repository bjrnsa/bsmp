# %%
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from bsmp.data_models.data_loader import MatchDataLoader
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

    NAME = "GSSD"

    def __init__(
        self,
        df: pd.DataFrame,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize GSSD model.

        Args:
            df (pd.DataFrame): DataFrame containing match data with columns:
                - home_team: Home team identifier
                - away_team: Away team identifier
                - home_goals: Points scored by home team
                - away_goals: Points scored by away team
                - goal_difference: home_goals - away_goals
            match_weights (np.ndarray, optional): Weights for match prediction.
                If None, equal weights are used.
        """
        # Extract numpy arrays once
        data = {
            col: df[col].to_numpy()
            for col in ["home_team", "away_team", "goal_difference"]
        }
        self.__dict__.update(data)

        # Team setup
        self.teams = np.unique(df.home_team.to_numpy())
        self.n_teams = len(self.teams)
        self.team_map = {team: idx for idx, team in enumerate(self.teams)}

        # Create team indices
        self.home_idx = np.array([self.team_map[team] for team in self.home_team])
        self.away_idx = np.array([self.team_map[team] for team in self.away_team])

        # Set weights
        self.ratings_weights = (
            np.ones(len(df)) if ratings_weights is None else ratings_weights
        )
        self.match_weights = (
            np.ones(len(df)) if match_weights is None else match_weights
        )

        # Calculate team statistics
        self._calculate_team_statistics(df)
        self.fitted = False

    def fit(self) -> "GSSD":
        """
        Fit the GSSD model to the data.

        Estimates team-specific coefficients using weighted OLS regression.
        Sets is_fitted to True upon completion.

        Returns:
            GSSD: The fitted GSSD model instance.
        """
        try:
            # Prepare features and fit model
            features = np.column_stack((self.pfh, self.pah, self.pfa, self.paa))
            initial_guess = np.array(
                [0.1, 1.0, 1.0, -1.0, -1.0]
            )  # [intercept, pfh, pah, pfa, paa]
            result = minimize(
                self._sse_function,
                initial_guess,
                method="L-BFGS-B",
                options={"ftol": 1e-10, "maxiter": 200},
            )

            # Store model parameters
            self.const = result.x[0]
            self.pfh_coeff = result.x[1]
            self.pah_coeff = result.x[2]
            self.pfa_coeff = result.x[3]
            self.paa_coeff = result.x[4]

            # Calculate spread error
            predictions = self._get_predictions(features)
            (self.intercept, self.spread_coefficient), self.spread_error = (
                self._fit_ols(self.goal_difference, predictions)
            )
            self.fitted = True
            return self

        except Exception as e:
            self.fitted = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def _sse_function(self, parameters: np.ndarray) -> float:
        """
        Calculate sum of squared errors for parameter optimization.

        Args:
            parameters (np.ndarray): Array of [intercept, pfh_coeff, pah_coeff, pfa_coeff, paa_coeff]

        Returns:
            float: Sum of squared errors
        """
        intercept, pfh_coeff, pah_coeff, pfa_coeff, paa_coeff = parameters

        # Vectorized calculation of predictions
        predictions = (
            intercept
            + pfh_coeff * self.pfh
            + pah_coeff * self.pah
            + pfa_coeff * self.pfa
            + paa_coeff * self.paa
        )

        # Calculate weighted squared errors
        errors = self.goal_difference - predictions
        sse = np.sum(self.ratings_weights * (errors**2))

        return sse

    def _get_predictions(self, features: np.ndarray) -> np.ndarray:
        """Calculate predictions using current model parameters.

        Args:
            features (np.ndarray): Feature matrix for predictions.

        Returns:
            np.ndarray: Predicted values.
        """
        return (
            self.const
            + self.pfh_coeff * features[:, 0]
            + self.pah_coeff * features[:, 1]
            + self.pfa_coeff * features[:, 2]
            + self.paa_coeff * features[:, 3]
        )

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
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            point_spread (float, optional): Point spread adjustment. Defaults to 0.0
            include_draw (bool, optional): Whether to include draw probability. Defaults to True

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
            self.const
            + home_off * self.pfh_coeff
            + home_def * self.pah_coeff
            + away_off * self.pfa_coeff
            + away_def * self.paa_coeff
        )

        predicted_spread = self.intercept + self.spread_coefficient * predicted_spread

        if return_spread:
            return predicted_spread

        return self._calculate_probabilities(
            predicted_spread=predicted_spread,
            std_error=self.spread_error,
            point_spread=point_spread,
            include_draw=include_draw,
        )

    def _calculate_probabilities(
        self,
        predicted_spread: float,
        std_error: float,
        point_spread: float = 0.0,
        include_draw: bool = True,
    ) -> Tuple[float, float, float]:
        """Calculate win/draw/loss probabilities using normal distribution.

        Args:
            predicted_spread (float): The predicted point spread.
            std_error (float): The standard error of the predictions.
            point_spread (float, optional): Point spread adjustment. Defaults to 0.0
            include_draw (bool, optional): Whether to include draw probability. Defaults to True

        Returns:
            Tuple[float, float, float]: Probabilities of (home win, draw, away win)
        """
        if include_draw:
            thresholds = np.array([point_spread + 0.5, -point_spread - 0.5])
            probs = stats.norm.cdf(thresholds, predicted_spread, std_error)
            return 1 - probs[0], probs[0] - probs[1], probs[1]
        else:
            prob_home = 1 - stats.norm.cdf(point_spread, predicted_spread, std_error)
            return prob_home, np.nan, 1 - prob_home

    def _calculate_team_statistics(self, df: pd.DataFrame) -> None:
        """
        Calculate and store all team-related statistics.

        Args:
            df (pd.DataFrame): DataFrame containing match data
        """
        # Calculate mean points for home/away scenarios
        home_stats = df.groupby("home_team").agg(
            {"home_goals": "mean", "away_goals": "mean"}
        )
        away_stats = df.groupby("away_team").agg(
            {"away_goals": "mean", "home_goals": "mean"}
        )

        # Store transformed statistics
        self.pfh = df.groupby("home_team")["home_goals"].transform("mean").to_numpy()
        self.pah = df.groupby("home_team")["away_goals"].transform("mean").to_numpy()
        self.pfa = df.groupby("away_team")["away_goals"].transform("mean").to_numpy()
        self.paa = df.groupby("away_team")["home_goals"].transform("mean").to_numpy()

        # Create team ratings dictionary
        self.team_ratings = {
            team: np.array(
                [
                    home_stats.loc[team, "home_goals"],
                    home_stats.loc[team, "away_goals"],
                    away_stats.loc[team, "away_goals"],
                    away_stats.loc[team, "home_goals"],
                ]
            )
            for team in self.teams
        }

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit weighted OLS regression using match weights.

        Args:
            y (np.ndarray): The dependent variable.
            X (np.ndarray): The independent variables.

        Returns:
            Tuple[np.ndarray, float]: Coefficients and standard error.
        """
        X = np.column_stack((np.ones(len(X)), X))
        W = np.diag(self.match_weights)

        # Use more efficient matrix operations
        XtW = X.T @ W
        coefficients = np.linalg.solve(XtW @ X, XtW @ y)
        residuals = y - X @ coefficients
        weighted_sse = residuals.T @ W @ residuals
        std_error = np.sqrt(weighted_sse / (len(y) - X.shape[1]))

        return coefficients, std_error

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        return pd.DataFrame(
            self.team_ratings, index=["pfh", "pah", "pfa", "paa"]
        ).T.round(2)

    def get_team_rating(self, team: str) -> float:
        """Get rating for a specific team."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        #  Return ratings first two decimals
        return np.round(self.team_ratings[team], 2)


if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )
    team_weights = dixon_coles_weights(df.datetime)

    home_team = "GOG"
    away_team = "Mors"

    model = GSSD(df, ratings_weights=team_weights)
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

# %%
