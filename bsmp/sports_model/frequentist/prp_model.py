# %%

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.zsd_model import ZSD
from bsmp.sports_model.utils import dixon_coles_weights


class PRP(ZSD):
    """
    Points Rating Prediction (PRP) model for predicting sports match outcomes with scikit-learn-like API.

    This model extends the ZSD model and estimates offensive and defensive ratings for each team,
    plus adjustment factors. The model uses a simple additive approach for score prediction:
    score = adj_factor + offense - defense + avg_score

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
    """

    NAME = "PRP"

    def __init__(self) -> None:
        """Initialize PRP model."""
        super().__init__()

    def _predict_scores(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
        offense_ratings: Union[np.ndarray, None] = None,
        defense_ratings: Union[np.ndarray, None] = None,
        factors: Union[Tuple[float, float], None] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate predicted scores using offensive/defensive ratings.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None], default=None
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None], default=None
            Index(es) of away team(s)
        offense_ratings : Union[np.ndarray, None], default=None
            Optional offensive ratings to use
        defense_ratings : Union[np.ndarray, None], default=None
            Optional defensive ratings to use
        factors : Union[Tuple[float, float], None], default=None
            Optional (home_factor, away_factor) tuple

        Returns
        -------
        Dict[str, np.ndarray]
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params_[-2:]

        ratings = self._get_team_ratings(
            home_idx, away_idx, offense_ratings, defense_ratings
        )

        return {
            "home": self._goal_func(
                factors[0],
                ratings["home_offense"],
                ratings["away_defense"],
                factors[1],
                factor=0.5,
            ),
            "away": self._goal_func(
                factors[0],
                ratings["away_offense"],
                ratings["home_defense"],
                factors[1],
                factor=-0.5,
            ),
        }

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None],
        away_idx: Union[int, np.ndarray, None],
        offense_ratings: Union[np.ndarray, None] = None,
        defense_ratings: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract offensive/defensive ratings from parameters.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None]
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None]
            Index(es) of away team(s)
        offense_ratings : Union[np.ndarray, None], default=None
            Optional offensive ratings to use
        defense_ratings : Union[np.ndarray, None], default=None
            Optional defensive ratings to use

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with team ratings
        """
        if offense_ratings is None:
            offense_ratings, defense_ratings = np.split(
                self.params_[: 2 * self.n_teams_], 2
            )
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

        return {
            "home_offense": offense_ratings[home_idx],
            "home_defense": defense_ratings[home_idx],
            "away_offense": offense_ratings[away_idx],
            "away_defense": defense_ratings[away_idx],
        }

    def _goal_func(
        self,
        home_advantage: float,
        offense_ratings: np.ndarray,
        defense_ratings: np.ndarray,
        avg_score: float,
        factor: float = 0.5,
    ) -> np.ndarray:
        """
        Calculate score prediction.

        Parameters
        ----------
        home_advantage : float
            Home advantage factor
        offense_ratings : np.ndarray
            Team's offensive ratings
        defense_ratings : np.ndarray
            Opponent's defensive ratings
        avg_score : float
            Average score factor
        factor : float, default=0.5
            Home/away adjustment multiplier

        Returns
        -------
        np.ndarray
            Predicted score
        """
        return (
            (factor * home_advantage) + (offense_ratings + defense_ratings) + avg_score
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Team ratings with columns ['team', 'offense', 'defense']
        """
        self._check_is_fitted()

        offense_ratings = self.params_[: self.n_teams_]
        defense_ratings = self.params_[self.n_teams_ : 2 * self.n_teams_]

        return pd.DataFrame(
            {
                "team": self.teams_,
                "offense": offense_ratings,
                "defense": defense_ratings,
            }
        ).set_index("team")

    def get_team_rating(self, team: str) -> Dict[str, float]:
        """
        Get ratings for a specific team.

        Parameters
        ----------
        team : str
            Name of the team

        Returns
        -------
        Dict[str, float]
            Dict with keys 'offense' and 'defense'
        """
        self._check_is_fitted()
        self._validate_teams([team])

        idx = self.team_map_[team]
        return {
            "offense": self.params_[idx],
            "defense": self.params_[idx + self.n_teams_],
        }


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

    home_team = "GOG"
    away_team = "Mors"

    # Create and fit the model
    model = PRP()

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
    predicted_spread = model.predict(X_test)
    print(f"Predicted goal difference: {predicted_spread[0]:.2f}")

    # Predict probabilities
    probs = model.predict_proba(X_test, point_spread=0, include_draw=True)

    print(f"Home win probability: {probs[0, 0]:.4f}")
    print(f"Draw probability: {probs[0, 1]:.4f}")
    print(f"Away win probability: {probs[0, 2]:.4f}")

    home_ratings = model.get_team_rating(home_team)
    away_ratings = model.get_team_rating(away_team)
    print(f"Home ratings: {home_ratings}")
    print(f"Away ratings: {away_ratings}")
