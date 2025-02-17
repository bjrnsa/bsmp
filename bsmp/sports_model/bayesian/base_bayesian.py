# %%
import os
from typing import Dict, Sequence, Union

import arviz as az
import cmdstanpy
import pandas as pd
from numpy.typing import NDArray


class BaseModel:
    """Base class for Bayesian Models"""

    def __init__(
        self,
        data: pd.DataFrame,
        weights: Union[float, Sequence[float], NDArray] = 1.0,
        stem: str = "base",
    ):
        """
        Initializes the BaseModel with data, weights, and model stem.

        Args:
            data (pd.DataFrame): The dataset containing match information.
            weights (Union[float, Sequence[float], NDArray], optional): Weights for the matches. Defaults to 1.0.
            stem (str, optional): The stem name for the Stan model file. Defaults to "base".
        """
        self.matches = data
        self.matches["weights"] = weights
        self.n_teams = len(
            set(self.matches["home_team"].unique())
            | set(self.matches["away_team"].unique())
        )

        self.model = None
        self.fit_result = None
        self.fitted = False

        self.STAN_FILE = os.path.join(
            os.path.dirname(__file__),
            "stan_files",
            f"{stem}.stan",
        )

    def _compile_and_fit_stan_model(
        self, stan_file: str, data: Dict, draws: int, warmup: int, chains: int
    ) -> cmdstanpy.CmdStanMCMC:
        """
        Compiles and fits the Stan model.

        Args:
            stan_file (str): The path to the Stan model file.
            data (dict): The data dictionary for the model.
            draws (int): Number of posterior draws.
            warmup (int): Number of warmup draws.
            chains (int): Number of Markov chains.

        Returns:
            cmdstanpy.CmdStanMCMC: The fit result object.
        """
        self.model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        self.fit_result = self.model.sample(
            data=data, iter_sampling=draws, iter_warmup=warmup, chains=chains
        )
        self.fitted = True
        self.n_samples = draws * chains
        return self.fit_result

    def _generate_inference_data(self):
        """
        Generates inference data using ArviZ.

        Returns:
            az.InferenceData: The inference data object.

        Raises:
            ValueError: If the model has not been fit yet.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")

        return az.from_cmdstanpy(
            posterior=self.fit_result,
            observed_data={
                "home_goals": self.matches["home_goals"].values,
                "away_goals": self.matches["away_goals"].values,
            },
            constant_data={
                "N": len(self.matches),
                "T": self.n_teams + 1,
                "home_team": self.matches["home_index"].values,
                "away_team": self.matches["away_index"].values,
                "weights": self.matches["weights"].values,
            },
            coords={
                "match": self.matches.index.tolist(),
                "team": self.matches["home_team"].unique().tolist(),
            },
            dims={
                "attack": ["team"],
                "attack_raw": ["team"],
                "defence": ["team"],
                "defence_raw": ["team"],
                "home_team": ["match"],
                "away_team": ["match"],
                "home_goals": ["match"],
                "away_goals": ["match"],
                "weights": ["match"],
            },
        )

    def _generate_posterior_data(self):
        """
        Generates posterior data from the inference data.

        Returns:
            az.InferenceData.posterior: The posterior data object.
        """
        return self._generate_inference_data().posterior

    def fit(self, draws: int, warmup: int):
        """
        Abstract method to fit the model. Must be implemented in subclasses.

        Args:
            draws (int): Number of posterior draws.
            warmup (int): Number of warmup draws.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The 'fit' method must be implemented in subclasses.")

    def predict(self, home_team: str, away_team: str, max_goals: int, n_samples: int):
        """
        Abstract method to predict match outcomes. Must be implemented in subclasses.

        Args:
            home_team (str): The home team name.
            away_team (str): The away team name.
            max_goals (int): The maximum number of goals to predict.
            n_samples (int): The number of samples to generate.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The 'predict' method must be implemented in subclasses."
        )


if __name__ == "__main__":
    pass
