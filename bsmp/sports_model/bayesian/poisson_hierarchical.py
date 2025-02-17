# %%

import numpy as np
import pandas as pd
from scipy.stats import poisson

from bsmp.sports_model.bayesian.base_bayesian import BaseModel
from bsmp.sports_model.bayesian.probability_grid import HandballProbabilityGrid


class PoissonHierarchical(BaseModel):
    """Bayesian Poisson Hierarchical Model"""

    def __init__(
        self,
        data: pd.DataFrame,
        weights: np.ndarray = 1.0,
        stem: str = "poisson_hierarchical",
    ):
        """
        Initializes the PoissonHierarchical model with data, weights, and model stem.

        Args:
            data (pd.DataFrame): The dataset containing match information.
            weights (np.ndarray, optional): Weights for the matches. Defaults to 1.0.
            stem (str, optional): The stem name for the Stan model file. Defaults to "poisson_hierarchical".
        """
        super().__init__(data, weights, stem)

    def fit(self, draws: int = 8000, warmup: int = 2000, chains: int = 6):
        """
        Fits the Bayesian Hierarchical Model.

        Args:
            draws (int, optional): Number of posterior draws to generate. Defaults to 8000.
            warmup (int, optional): Number of warmup draws. Defaults to 2000.
            chains (int, optional): Number of Markov chains. Defaults to 6.
        """
        data = {
            "N": len(self.matches),
            "T": self.n_teams,
            "home_goals": self.matches["home_goals"].values,
            "away_goals": self.matches["away_goals"].values,
            "home_team": self.matches["home_index"].values,
            "away_team": self.matches["away_index"].values,
            "weights": self.matches["weights"].values,
        }

        self._compile_and_fit_stan_model(self.STAN_FILE, data, draws, warmup, chains)

        return self

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 90, n_samples: int = None
    ) -> HandballProbabilityGrid:
        """
        Predicts the probability of goals scored by a home team and an away team.

        Args:
            home_team (str): The name of the home team.
            away_team (str): The name of the away team.
            max_goals (int, optional): The maximum number of goals to consider. Defaults to 90.
            n_samples (int, optional): The number of samples to use for prediction. Defaults to None.

        Returns:
            HandballProbabilityGrid: A HandballProbabilityGrid object containing the predicted probabilities.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")

        draws = self._generate_posterior_data()
        attack = draws.attack
        defence = draws.defence
        home_adv = draws.home_advantage.values.flatten()
        goal_mean = draws.goal_mean.values.flatten()

        lambda_home = np.exp(
            goal_mean
            + attack.sel(team=home_team).values.flatten()
            - defence.sel(team=away_team).values.flatten()
            + home_adv
        )
        lambda_away = np.exp(
            goal_mean
            + attack.sel(team=away_team).values.flatten()
            - defence.sel(team=home_team).values.flatten()
        )

        if n_samples is None:
            n_samples = self.n_samples
            lambda_away = np.random.choice(lambda_away, size=n_samples)
            lambda_home = np.random.choice(lambda_home, size=n_samples)

        home_probs = poisson.pmf(np.arange(max_goals + 1)[:, None], lambda_home)
        away_probs = poisson.pmf(
            np.arange(max_goals + 1)[None, :], lambda_away[:, None]
        )

        # Numerical stability crucial for handball's high scores
        score_probs = np.tensordot(home_probs, away_probs, axes=(1, 0)) / n_samples
        score_probs = np.nan_to_num(score_probs, nan=0.0, posinf=0.0, neginf=0.0)
        score_probs /= score_probs.sum()

        # Expectation calculation remains same but yields higher values
        home_expectancy = (score_probs.sum(axis=1) * np.arange(max_goals + 1)).sum()
        away_expectancy = (score_probs.sum(axis=0) * np.arange(max_goals + 1)).sum()

        return HandballProbabilityGrid(
            goal_matrix=score_probs,
            home_goal_expectation=home_expectancy,
            away_goal_expectation=away_expectancy,
            model_class=self.__class__.__name__,
        )


if __name__ == "__main__":
    pass
