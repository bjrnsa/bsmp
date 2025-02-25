# %%

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.zsd_model import ZSD
from bsmp.sports_model.utils import dixon_coles_weights


class PRP(ZSD):
    """
    Points Rating Prediction (PRP) model for predicting sports match outcomes.

    This model estimates offensive and defensive ratings for each team, plus
    adjustment factors. Parameter structure:
    - Offensive ratings for each team
    - Defensive ratings for each team
    - Home/away adjustment factors

    Score prediction uses a simple additive model:
    score = adj_factor + offense - defense + avg_score
    """

    NAME = "PRP"

    def __init__(
        self,
        df,
        ratings_weights: Union[np.ndarray, None] = None,
        match_weights: Union[np.ndarray, None] = None,
    ):
        super().__init__(df, ratings_weights, match_weights)

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

        Args:
            home_idx: Index(es) of home team(s)
            away_idx: Index(es) of away team(s)
            offense_ratings: Optional offensive ratings to use
            defense_ratings: Optional defensive ratings to use
            factors: Optional (home_factor, away_factor) tuple

        Returns:
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params[-2:]

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
        """Extract offensive/defensive ratings from parameters."""
        if offense_ratings is None:
            offense_ratings, defense_ratings = np.split(
                self.params[: 2 * self.n_teams], 2
            )
        if home_idx is None:
            home_idx, away_idx = self.home_idx, self.away_idx

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

        Args:
            home_advantage: Home advantage factor
            offense_ratings: Team's offensive ratings
            defense_ratings: Opponent's defensive ratings
            avg_score: Average score factor
            factor: Home/away adjustment multiplier

        Returns:
            Predicted score
        """
        return (
            (factor * home_advantage) + (offense_ratings + defense_ratings) + avg_score
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """
        Get team ratings as a DataFrame.

        Returns:
            pd.DataFrame: Team ratings with columns ['team', 'offense', 'defense']

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        offense_ratings = self.params[: self.n_teams]
        defense_ratings = self.params[self.n_teams : 2 * self.n_teams]

        return pd.DataFrame(
            {"team": self.teams, "offense": offense_ratings, "defense": defense_ratings}
        ).set_index("team")

    def get_team_rating(self, team: str) -> float:
        """
        Get ratings for a specific team.

        Args:
            team: Name of the team

        Returns:
            Dict with keys 'offense' and 'defense'

        Raises:
            ValueError: If team not found or model not fitted
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        if team not in self.team_map:
            raise ValueError(f"Unknown team: {team}")

        idx = self.team_map[team]
        return {"offense": self.params[idx], "defense": self.params[idx + self.n_teams]}


if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )
    team_weights = dixon_coles_weights(df.datetime)

    home_team = "GOG"
    away_team = "Mors"

    model = PRP(df, ratings_weights=team_weights)
    model.fit()
    prob_home, prob_draw, prob_away = model.predict(
        home_team, away_team, point_spread=2, include_draw=True
    )
    print(f"Home win probability: {prob_home}")
    print(f"Draw probability: {prob_draw}")
    print(f"Away win probability: {prob_away}")
    print(f"Home ratings: {model.get_team_rating(home_team)}")
    print(f"Away ratings: {model.get_team_rating(away_team)}")
