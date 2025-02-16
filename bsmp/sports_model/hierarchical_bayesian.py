# %%
import os
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.base_bayesian import BaseModel


class HierarchicalModel(BaseModel):
    """Bayesian Hierarchical Model"""

    STAN_FILE = os.path.join(
        os.path.dirname(__file__),
        "stan_files",
        "hierarchical_bayesian.stan",
    )

    def _get_model_parameters(self):
        draws = self.fit_result.draws_pd()
        att_params = [x for x in draws.columns if "attack" in x]
        defs_params = [x for x in draws.columns if "defence" in x]
        return draws, att_params, defs_params

    def _format_team_parameters(self, draws, att_params, defs_params):
        attack = [None] * self.T
        defence = [None] * self.T
        team = self.teams["team"].tolist()

        atts = draws[att_params].mean()
        defs = draws[defs_params].mean()

        for idx, _ in enumerate(team):
            attack[idx] = round(atts.iloc[idx], 3)
            defence[idx] = round(defs.iloc[idx], 3)

        return team, attack, defence

    def __repr__(self):
        repr_str = "Module: Penaltyblog\n\nModel: Bayesian Bivariate (Stan)\n\n"

        if not self.fitted:
            return repr_str + "Status: Model not fitted"

        draws, att_params, defs_params = self._get_model_parameters()
        team, attack, defence = self._format_team_parameters(
            draws, att_params, defs_params
        )

        repr_str += f"Number of parameters: {len(att_params) + len(defs_params) + 2}\n"
        repr_str += "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defence")
        repr_str += "\n" + "-" * 60 + "\n"

        for t, a, d in zip(team, attack, defence):
            repr_str += "{0: <20} {1:<20} {2:<20}\n".format(t, a, d)

        repr_str += "-" * 60 + "\n"
        repr_str += f"Home Advantage: {round(draws['home'].mean(), 3)}\n"
        # repr_str += f"Intercept: {round(draws['intercept'].mean(), 3)}\n"

        return repr_str

    def fit(self, draws: int = 8000, warmup: int = 2000, chains: int = 6):
        """
        Fits the Bayesian Hierarchical Model.

        Args:
            draws (int, optional): Number of posterior draws to generate, defaults to 5000.
            warmup (int, optional): Number of warmup draws, defaults to 2000.
        """
        data = {
            "N": len(self.fixtures),
            "T": len(self.teams),
            "home_goals": self.fixtures["home_goals"].values,
            "away_goals": self.fixtures["away_goals"].values,
            "home_team": self.fixtures["home_index"].values,
            "away_team": self.fixtures["away_index"].values,
            "weights": self.fixtures["weights"].values,
        }

        self._compile_and_fit_stan_model(self.STAN_FILE, data, draws, warmup, chains)
        self.iference_data()
        return self

    def iference_data(self):
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")
        coords = {
            "team": self.teams["team"].tolist(),
            "match": np.arange(len(self.fixtures)),
        }

        dims = {
            "att": ["team"],
            "def": ["team"],
            "home_team": ["match"],
            "away_team": ["match"],
            "home_goals": ["match"],
            "away_goals": ["match"],
        }

        observed_data = {
            "home_goals": self.fixtures["home_goals"].values,
            "away_goals": self.fixtures["away_goals"].values,
        }

        constant_data = {
            "home_team": self.fixtures["home_index"].values,
            "away_team": self.fixtures["away_index"].values,
        }

        self.inference_data = az.from_cmdstanpy(
            posterior=self.fit_result,
            observed_data=observed_data,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
        )

    def get_params(self) -> Dict:
        """
        Returns the fitted parameters of the Bayesian Bivariate Goal Model.

        Returns:
            dict: A dictionary containing the fitted parameters of the model.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before getting parameters")

        posterior = self.inference_data.posterior
        team_stats = (
            posterior[["att", "def"]].mean(dim=["chain", "draw"]).to_dataframe()
        ).round(3)
        home_adv = posterior["home_advantage"].mean(dim=["chain", "draw"]).round(3)
        goal_mean = posterior["goal_mean"].mean(dim=["chain", "draw"]).round(3)

        return team_stats, home_adv, goal_mean

    def predict(
        self, home_team: str, away_team: str, n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Predicts the probability of goals scored by a home team and an away team.

        Args:
            home_team (str): The name of the home team.
            away_team (str): The name of the away team.
            max_goals (int, optional): The maximum number of goals to consider, defaults to 15.
            n_samples (int, optional): The number of samples to use for prediction, defaults to 1000.

        Returns:
                FootballProbabilityGrid: A FootballProbabilityGrid object containing
                the predicted probabilities.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")

        att = self.inference_data.posterior["att"].to_dataframe().reset_index(["team"])
        def_ = self.inference_data.posterior["def"].to_dataframe().reset_index(["team"])
        home_adv = self.inference_data.posterior["home_advantage"].to_dataframe()
        goal_mean = self.inference_data.posterior["goal_mean"].to_dataframe()

        lambda_home = np.exp(
            goal_mean["goal_mean"]
            + att.query("team == @home_team")["att"]
            - def_.query("team == @away_team")["def"]
            + home_adv["home_advantage"]
        )
        lambda_away = np.exp(
            goal_mean["goal_mean"]
            + att.query("team == @away_team")["att"]
            - def_.query("team == @home_team")["def"]
        )

        home_goals = np.random.poisson(lambda_home)
        away_goals = np.random.poisson(lambda_away)

        results = pd.DataFrame(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_team": home_team,
                "away_team": away_team,
            }
        )

        return self._calculate_metrics(results)


if __name__ == "__main__":
    # Load match data
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        season="2024/2025",
        league="Herre Handbold Ligaen",
        date_range=("2024-08-01", "2025-02-01"),
        team_filters={"country": "Denmark"},
    )

    # Fit Bayesian       Goal Model
    model = HierarchicalModel(
        home_goals=df["hg"],
        away_goals=df["ag"],
        home_team=df["home_team"],
        away_team=df["away_team"],
    )
    model.fit()

    team_1 = np.random.choice(df["home_team"].unique())
    team_2 = np.random.choice(df["away_team"].unique())

    # Predict match outcome
    prediction = model.predict(home_team=team_1, away_team=team_2)
    print(prediction)
    # %%
    # Get model parameters
    # Get model parameters
    # Get model parameters
