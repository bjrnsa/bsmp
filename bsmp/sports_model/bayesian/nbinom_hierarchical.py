# %%

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from bsmp.sports_model.bayesian.poisson_hierarchical import PoissonHierarchical
from bsmp.sports_model.bayesian.probability_grid import HandballProbabilityGrid


class NbinomHierarchical(PoissonHierarchical):
    """Bayesian Negative Binomial Hierarchical Model"""

    def __init__(
        self,
        data: pd.DataFrame,
        weights: np.ndarray = 1.0,
        stem: str = "nbinom_hierarchical",
    ):
        """
        Initializes the NbinomHierarchical model with data, weights, and model stem.

        Args:
            data (pd.DataFrame): The dataset containing match information.
            weights (np.ndarray, optional): Weights for the matches. Defaults to 1.0.
            stem (str, optional): The stem name for the Stan model file. Defaults to "nbinom_hierarchical".
        """
        super().__init__(data, weights, stem)

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
        theta = draws.theta.values.flatten()

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

        ranged_ = np.arange(max_goals + 1)[:, None]

        home_probs = nbinom.pmf(
            ranged_,
            n=theta,
            p=theta / (theta + lambda_home),
        )

        away_probs = nbinom.pmf(
            ranged_,
            n=theta,
            p=(theta / (theta + lambda_away)),
        ).T

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
