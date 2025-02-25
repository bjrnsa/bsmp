# %%
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.utils import dixon_coles_weights


class GSSD:
    """
    Generalized Scores Standard Deviation (GSSD) model with scikit-learn-like API.

    A model that predicts match outcomes using team-specific offensive and defensive ratings.
    The model uses weighted OLS regression to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

    Parameters
    ----------
    None

    Attributes
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    team_ratings_ : Dict[str, np.ndarray]
        Dictionary mapping teams to their offensive/defensive ratings
    is_fitted_ : bool
        Whether the model has been fitted
    spread_error_ : float
        Standard error of the model predictions
    intercept_ : float
        Model intercept term
    pfh_coeff_ : float
        Coefficient for home team's offensive rating
    pah_coeff_ : float
        Coefficient for home team's defensive rating
    pfa_coeff_ : float
        Coefficient for away team's offensive rating
    paa_coeff_ : float
        Coefficient for away team's defensive rating
    """

    NAME = "GSSD"

    def __init__(self) -> None:
        """Initialize GSSD model."""
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: pd.DataFrame = None,
        ratings_weights: Optional[np.ndarray] = None,
        match_weights: Optional[np.ndarray] = None,
    ) -> "GSSD":
        """
        Fit the GSSD model to the data.

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

        Returns
        -------
        self : GSSD
            The fitted GSSD model instance.
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

            # Extract home_goals and away_goals from Z
            self.home_goals_ = Z.iloc[:, 0].to_numpy()
            self.away_goals_ = Z.iloc[:, 1].to_numpy()

            # Team setup
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

            # Calculate team statistics
            self._calculate_team_statistics(X)

            # Prepare features and fit model
            features = np.column_stack((self.pfh_, self.pah_, self.pfa_, self.paa_))
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
            self.const_ = result.x[0]
            self.pfh_coeff_ = result.x[1]
            self.pah_coeff_ = result.x[2]
            self.pfa_coeff_ = result.x[3]
            self.paa_coeff_ = result.x[4]

            # Calculate spread error
            predictions = self._get_predictions(features)
            (self.intercept_, self.spread_coefficient_), self.spread_error_ = (
                self._fit_ols(self.goal_difference_, predictions)
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

            # Get team ratings
            home_off, home_def = self.team_ratings_[home_team][:2]
            away_off, away_def = self.team_ratings_[away_team][2:]

            # Calculate spread
            predicted_spread = (
                self.const_
                + home_off * self.pfh_coeff_
                + home_def * self.pah_coeff_
                + away_off * self.pfa_coeff_
                + away_def * self.paa_coeff_
            )

            predicted_spreads[i] = (
                self.intercept_ + self.spread_coefficient_ * predicted_spread
            )

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

            # Get team ratings
            home_off, home_def = self.team_ratings_[home_team][:2]
            away_off, away_def = self.team_ratings_[away_team][2:]

            # Calculate spread
            predicted_spread = (
                self.const_
                + home_off * self.pfh_coeff_
                + home_def * self.pah_coeff_
                + away_off * self.pfa_coeff_
                + away_def * self.paa_coeff_
            )

            predicted_spread = (
                self.intercept_ + self.spread_coefficient_ * predicted_spread
            )

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

    def _sse_function(self, parameters: np.ndarray) -> float:
        """
        Calculate sum of squared errors for parameter optimization.

        Parameters
        ----------
        parameters : np.ndarray
            Array of [intercept, pfh_coeff, pah_coeff, pfa_coeff, paa_coeff]

        Returns
        -------
        float
            Sum of squared errors
        """
        intercept, pfh_coeff, pah_coeff, pfa_coeff, paa_coeff = parameters

        # Vectorized calculation of predictions
        predictions = (
            intercept
            + pfh_coeff * self.pfh_
            + pah_coeff * self.pah_
            + pfa_coeff * self.pfa_
            + paa_coeff * self.paa_
        )

        # Calculate weighted squared errors
        errors = self.goal_difference_ - predictions
        sse = np.sum(self.ratings_weights_ * (errors**2))

        return sse

    def _get_predictions(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate predictions using current model parameters.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix for predictions.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return (
            self.const_
            + self.pfh_coeff_ * features[:, 0]
            + self.pah_coeff_ * features[:, 1]
            + self.pfa_coeff_ * features[:, 2]
            + self.paa_coeff_ * features[:, 3]
        )

    def _calculate_team_statistics(self, X: pd.DataFrame) -> None:
        """
        Calculate and store all team-related statistics.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data
        """
        # Create a temporary DataFrame for calculations
        df = pd.DataFrame(
            {
                "home_team": self.home_team_,
                "away_team": self.away_team_,
                "home_goals": self.home_goals_,
                "away_goals": self.away_goals_,
            }
        )

        # Calculate mean points for home/away scenarios
        home_stats = df.groupby("home_team").agg(
            {"home_goals": "mean", "away_goals": "mean"}
        )
        away_stats = df.groupby("away_team").agg(
            {"away_goals": "mean", "home_goals": "mean"}
        )

        # Store transformed statistics
        self.pfh_ = df.groupby("home_team")["home_goals"].transform("mean").to_numpy()
        self.pah_ = df.groupby("home_team")["away_goals"].transform("mean").to_numpy()
        self.pfa_ = df.groupby("away_team")["away_goals"].transform("mean").to_numpy()
        self.paa_ = df.groupby("away_team")["home_goals"].transform("mean").to_numpy()

        # Create team ratings dictionary
        self.team_ratings_ = {}
        for team in self.teams_:
            if team in home_stats.index and team in away_stats.index:
                self.team_ratings_[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            elif team in home_stats.index:
                # Team only played home games
                self.team_ratings_[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        0.0,  # Default value for missing away offensive rating
                        0.0,  # Default value for missing away defensive rating
                    ]
                )
            elif team in away_stats.index:
                # Team only played away games
                self.team_ratings_[team] = np.array(
                    [
                        0.0,  # Default value for missing home offensive rating
                        0.0,  # Default value for missing home defensive rating
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            else:
                # Team hasn't played any games (shouldn't happen with proper data)
                self.team_ratings_[team] = np.zeros(4)

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit weighted OLS regression using match weights.

        Parameters
        ----------
        y : np.ndarray
            The dependent variable.
        X : np.ndarray
            The independent variables.

        Returns
        -------
        Tuple[np.ndarray, float]
            Coefficients and standard error.
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
            if team not in self.team_ratings_:
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
            DataFrame with team ratings
        """
        self._check_is_fitted()
        return pd.DataFrame(
            self.team_ratings_, index=["pfh", "pah", "pfa", "paa"]
        ).T.round(2)

    def get_team_rating(self, team: str) -> np.ndarray:
        """
        Get rating for a specific team.

        Parameters
        ----------
        team : str
            Team name

        Returns
        -------
        np.ndarray
            Team ratings [pfh, pah, pfa, paa]
        """
        self._check_is_fitted()
        self._validate_teams([team])
        return np.round(self.team_ratings_[team], 2)


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
    team_weights = dixon_coles_weights(train_df.datetime)

    home_team = "Holstebro"
    away_team = "Skjern"

    # train_df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv").iloc[:177]
    # train_df = train_df.assign(
    #     goal_difference=train_df["Home Pts"] - train_df["Away Pts"]
    # )
    # train_df = train_df.assign(
    #     home_team=train_df["Home Team"], away_team=train_df["Away Team"]
    # )
    # train_df = train_df.assign(
    #     home_goals=train_df["Home Pts"], away_goals=train_df["Away Pts"]
    # )
    # home_team = "St Kilda"
    # away_team = "North Melbourne"

    # Create and fit the model
    model = GSSD()

    # Prepare training data
    X_train = train_df[["home_team", "away_team", "home_goals", "away_goals"]]
    y_train = train_df["goal_difference"]
    model.fit(X_train, y_train)

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
    print(f"Home rating: {model.get_team_rating(home_team)}")
    print(f"Away rating: {model.get_team_rating(away_team)}")

# %%
