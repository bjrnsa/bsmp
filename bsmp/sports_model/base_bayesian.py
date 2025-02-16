from typing import Dict, Sequence, Union

import cmdstanpy
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bsmp.data_models.data_loader import MatchDataLoader


class BaseModel:
    """Base class for Bayesian Models"""

    def __init__(
        self,
        home_goals: Union[Sequence[int], NDArray],
        away_goals: Union[Sequence[int], NDArray],
        home_team: Union[Sequence[int], NDArray],
        away_team: Union[Sequence[int], NDArray],
        weights: Union[float, Sequence[float], NDArray] = 1.0,
    ):
        self.fixtures = pd.DataFrame(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_team": home_team,
                "away_team": away_team,
                "weights": weights,
            }
        )
        self._setup_teams()
        self.model = None
        self.fit_result = None
        self.fitted = False

    def _setup_teams(self):
        unique_teams = pd.DataFrame(
            {
                "team": pd.concat(
                    [self.fixtures["home_team"], self.fixtures["away_team"]]
                ).unique()
            }
        )
        unique_teams = (
            unique_teams.sort_values("team")
            .reset_index(drop=True)
            .assign(team_index=lambda x: np.arange(len(x)) + 1)
        )

        self.n_teams = len(unique_teams)
        self.teams = unique_teams
        self.fixtures = (
            self.fixtures.merge(unique_teams, left_on="home_team", right_on="team")
            .rename(columns={"team_index": "home_index"})
            .drop("team", axis=1)
            .merge(unique_teams, left_on="away_team", right_on="team")
            .rename(columns={"team_index": "away_index"})
            .drop("team", axis=1)
        )

    def _compile_and_fit_stan_model(
        self, stan_file: str, data: Dict, draws: int, warmup: int, chains: int
    ) -> cmdstanpy.CmdStanMCMC:
        """
        Compiles and fits the Stan model.

        Args:
            stan_model (str): The Stan model code as a string.
            data (dict): The data dictionary for the model.
            draws (int): Number of posterior draws.
            warmup (int): Number of warmup draws.

        Returns:
            cmdstanpy.CmdStanMCMC: The fit result object.
        """
        self.model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        self.fit_result = self.model.sample(
            data=data, iter_sampling=draws, iter_warmup=warmup, chains=chains
        )
        self.fitted = True
        return self.fit_result

    def _calculate_metrics(self, match_results: pd.DataFrame) -> pd.Series:
        """
        Calculate match outcome probabilities and metrics.

        Args:
            match_results (pd.DataFrame): DataFrame containing the simulated match outcomes.

        Returns:
            dict: A dictionary containing the match outcome probabilities and metrics.
        """
        home_win_prob = np.mean(
            match_results["home_goals"] > match_results["away_goals"]
        )
        away_win_prob = np.mean(
            match_results["away_goals"] > match_results["home_goals"]
        )
        draw_prob = np.mean(match_results["home_goals"] == match_results["away_goals"])

        home_avg_goals = match_results["home_goals"].mean()
        away_avg_goals = match_results["away_goals"].mean()

        metrics = pd.Series(
            {
                "home_win_prob": home_win_prob,
                "away_win_prob": away_win_prob,
                "draw_prob": draw_prob,
                "home_avg_goals": home_avg_goals,
                "away_avg_goals": away_avg_goals,
                "home_win_odds": 1 / home_win_prob,
                "away_win_odds": 1 / away_win_prob,
                "draw_odds": 1 / draw_prob,
            }
        )

        return metrics.round(3)

    def fit(self, draws: int, warmup: int):
        raise NotImplementedError("The 'fit' method must be implemented in subclasses.")

    def predict(self, home_team: str, away_team: str, max_goals: int, n_samples: int):
        raise NotImplementedError(
            "The 'predict' method must be implemented in subclasses."
        )

    def __repr__(self):
        raise NotImplementedError(
            "The '__repr__' method must be implemented in subclasses."
        )

    def _get_team_index(self, team_name):
        return self.teams.loc[self.teams["team"] == team_name, "team_index"].iloc[0]


if __name__ == "__main__":
    df = MatchDataLoader(sport="handball").load_matches(
        season="2024/2025",
        league="Herre Handbold Ligaen",
        date_range=("2024-08-01", "2025-02-01"),
        team_filters={"country": "Denmark"},
    )
    model = BaseModel(
        home_goals=df["hg"].values,
        away_goals=df["ag"].values,
        home_team=df["home_index"].values,
        away_team=df["away_index"].values,
    )
