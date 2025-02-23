# %%
import numpy as np
import pandas as pd

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry


class TOOR(BradleyTerry):
    """
    TOOR (Team OLS Optimized Rating) model that extends Bradley-Terry.
    This model combines Bradley-Terry ratings with linear regression to predict
    match outcomes and point spreads using team-specific coefficients.
    """

    def __init__(
        self, df: pd.DataFrame, weights: np.ndarray = None, home_advantage: float = 0.1
    ):
        """
        Initialize TOOR model.

        Args:
            df: DataFrame containing match data
            weights: Optional weights for each observation
            home_advantage: Home field advantage parameter
        """
        super().__init__(df, weights, home_advantage)
        self.team_coefficients = None
        self.home_coefficient = None

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
    # Example usage
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(league="Herre Handbold Ligaen")

    # Load AFL data for testing
    df = pd.read_csv("bsmp/sports_model/frequentist/afl_data.csv").loc[:176]
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["game_total"] = df["away_pts"] + df["home_pts"]
    df["goal_difference"] = df["home_pts"] - df["away_pts"]
    df["result"] = np.where(df["goal_difference"] > 0, 1, -1)

    # Fit model
    model = TOOR(df, None, 0.1)
    model.fit()

    # Make predictions
    teams = [("St Kilda", "North Melbourne")]

    for home_team, away_team in teams:
        probs = model.predict(home_team, away_team, point_spread=0, include_draw=False)
        print(f"\n{home_team} vs {away_team}")
        print(f"Win probabilities (H/D/A): {probs}")
