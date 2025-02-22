# %%

import numpy as np
import scipy.stats as stats
from scipy.optimize import approx_fprime, minimize

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.utils import dixon_coles_weights


class BradleyTerryModel:
    def __init__(self, df, weights, hfa: float = 0.1):
        for col in df:
            setattr(self, col, df[col].to_numpy())

        self.teams = np.unique(self.home_team)
        self.team_ratings = dict.fromkeys(self.teams, 1.0)
        self.params = np.array(list(self.team_ratings.values()) + [hfa])
        self.data = np.array([self.home_team, self.away_team, self.result]).T
        self.nobs = len(self.data)
        self.weights = np.ones(self.nobs) if weights is None else weights

    def fit(self) -> None:
        """
        Fit the Bradley-Terry model to the data.

        Args:
            df (pd.DataFrame): DataFrame containing match data
            weights (np.ndarray, optional): Weights for each observation. Defaults to None.
        """

        # Store optimized parameters
        self.best_params = self.bradley_terry_params()
        self.team_ratings = dict(zip(self.teams, self.best_params[:-1]))
        self.hfa = self.best_params[-1]

        # Fit regression model for goal difference
        home_ratings = np.array([self.team_ratings[team] for team in self.home_team])
        away_ratings = np.array([self.team_ratings[team] for team in self.away_team])
        logit_scale = self._logit_transform(self.hfa + home_ratings - away_ratings)
        goal_difference = self.goal_difference

        (self.const, self.beta), self.std_error = self._ols(
            goal_difference, logit_scale
        )

    def predict(self, home_team: str, away_team: str, point_spread: float = 0.0):
        """
        Predict the probabilities of home win, draw, and away win.

        Args:
            home_team (str): Home team name
            away_team (str): Away team name
            point_spread (float, optional): Point spread. Defaults to 0.0.

        Returns:
            tuple: Probabilities of home win, draw, and away win
        """
        logit_val = self._logit_strength(home_team, away_team)
        pred = self.const + self.beta * logit_val

        prob_home = 1 - stats.norm.cdf(point_spread + 0.5, pred, self.std_error)
        prob_away = stats.norm.cdf(-point_spread - 0.5, pred, self.std_error)
        prob_draw = 1 - prob_home - prob_away

        return prob_home, prob_draw, prob_away

    def bradley_terry_params(self):
        """
        Optimize model parameters using the SLSQP algorithm.

        Returns:
            np.ndarray: Optimized parameters
        """
        result = minimize(
            lambda p: self._log_likelihood(p, self.data, self.weights) / self.nobs,
            self.params,
            jac=lambda p: approx_fprime(
                p,
                lambda x: self._log_likelihood(x, self.data, self.weights) / self.nobs,
                6.5e-07,
            ),
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )

        # Store optimized parameters
        self.best_params = result.x
        self.team_ratings = dict(zip(self.teams, self.best_params[:-1]))
        self.hfa = self.best_params[-1]
        return result.x

    def _log_likelihood(
        self, params: np.ndarray, data: np.ndarray, weights: np.ndarray
    ) -> float:
        """
        Calculate negative log likelihood for parameter optimization.
        Corrects for draw outcomes by splitting them into two separate outcomes.

        Args:
            params (np.ndarray): Model parameters
            data (np.ndarray): Training data
            weights (np.ndarray): Weights for each observation

        Returns:
            float: Negative log likelihood
        """
        ratings = params[:-1]
        hfa = params[-1]
        log_likelihood = 0.0

        for i, team in enumerate(data):
            ht = ratings[self.teams == team[0]][0]
            at = ratings[self.teams == team[1]][0]

            logistic = 1 / (1 + np.exp(-(hfa + ht - at)))
            if data[i][2] == 1:
                result = logistic
            elif data[i][2] == -1:
                result = 1 - logistic
            else:
                result_home = logistic
                result_away = 1 - logistic
                log_likelihood += (np.log(result_home) + np.log(result_away)) * weights[
                    i
                ]
                continue
            log_likelihood += np.log(result) * weights[i]

        return -log_likelihood

    def _logit_transform(self, x: float) -> float:
        """
        Apply logistic transformation.

        Args:
            x (float): Input value

        Returns:
            float: Transformed value
        """
        return 1 / (1 + np.exp(-x))

    def _logit_strength(self, home_team: str, away_team: str) -> float:
        """
        Calculate logit strength difference between teams.

        Args:
            home_team (str): Home team name
            away_team (str): Away team name

        Returns:
            float: Logit strength difference
        """
        return self._logit_transform(
            -(self.hfa + self.team_ratings[home_team] - self.team_ratings[away_team])
        )

    def _ols(self, y, X):
        """
        Perform Ordinary Least Squares (OLS) regression.

        Args:
            y (np.ndarray): Dependent variable
            X (np.ndarray): Independent variable(s)

        Returns:
            tuple: Parameters and residual standard error
        """
        X = np.column_stack((np.ones(len(X)), X))
        betas = np.linalg.inv(X.T @ X) @ X.T @ y
        residuals = y - X @ betas
        rse = np.sqrt(np.sum(residuals**2) / (len(y) - X.shape[1]))
        return betas, rse


if __name__ == "__main__":
    # Initialize and fit the model
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(league="Herre Handbold Ligaen")
    odds_df = loader.load_odds(df.index.unique())
    weights = dixon_coles_weights(df.datetime, xi=0.18)

    model = BradleyTerryModel(df, weights, 0.1)
    model.fit()

    # Make predictions
    home_team = "Kolding"
    away_team = "Bjerringbro/Silkeborg"
    prob_home, prob_draw, prob_away = model.predict(home_team, away_team)
