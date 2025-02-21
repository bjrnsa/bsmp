from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import approx_fprime, minimize

from bsmp.data_models.data_loader import MatchDataLoader


class BradleyTerryModel:
    def __init__(self, initial_hfa: float = 0.1):
        """
        Initialize Bradley-Terry Model for sports prediction.

        Args:
            initial_hfa (float, optional): Initial home field advantage value. Defaults to 0.1.
        """
        self.hfa = initial_hfa
        self.team_ratings = None
        self.teams = None
        self.params = None
        self.reg_params = None
        self.std_error = None

    def fit(self, df: pd.DataFrame, test_season: int = 2025) -> None:
        """
        Fit the Bradley-Terry model to the data.

        Args:
            df (pd.DataFrame): DataFrame containing match data
            test_season (int, optional): Season to use as test set. Defaults to 2025.
        """
        self.teams = df["home_team"].unique()
        self.team_ratings = dict.fromkeys(self.teams, 1.0)
        self.params = np.array(list(self.team_ratings.values()) + [self.hfa])

        # Prepare training data
        insample_data = df.query(f"season_year != {test_season}")[
            ["home_team", "away_team", "result"]
        ].values
        nobs = len(insample_data)

        # Optimize parameters
        result = minimize(
            lambda p: self._log_likelihood(p, insample_data) / nobs,
            self.params,
            jac=lambda p: approx_fprime(
                p, lambda x: self._log_likelihood(x, insample_data) / nobs, 6.5e-07
            ),
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )

        # Store optimized parameters
        best_params = result.x
        self.team_ratings = dict(zip(self.teams, best_params[:-1]))
        self.hfa = best_params[-1]

        # Fit regression model for goal difference
        df["home_rating"] = df["home_team"].map(self.team_ratings)
        df["away_rating"] = df["away_team"].map(self.team_ratings)
        df["logit_scale"] = df.apply(
            lambda x: self._logit_transform(
                self.hfa + x["home_rating"] - x["away_rating"]
            ),
            axis=1,
        )

        oos_df = df.query(f"season_year == {test_season}")
        model = sm.OLS(
            oos_df["goal_difference"], sm.add_constant(oos_df["logit_scale"])
        )
        reg = model.fit()

        self.reg_params = {
            "intercept": reg.params["const"],
            "slope": reg.params["logit_scale"],
        }
        self.std_error = reg.resid.std()

    def predict(
        self, home_team: str, away_team: str, point_spread: float = 1.0
    ) -> Dict:
        """
        Predict match outcome and calculate betting metrics.

        Args:
            home_team (str): Name of home team
            away_team (str): Name of away team
            point_spread (float, optional): Point spread for victory. Defaults to 1.0.

        Returns:
            Dict: Dictionary containing predictions and betting metrics
        """
        logit_val = self._logit_strength(home_team, away_team)
        predicted_mov = self.reg_params["intercept"] + self.reg_params["slope"] * (
            1 - logit_val
        )

        probability_home = 1 - stats.norm.cdf(
            point_spread, predicted_mov, self.std_error
        )
        probability_away = 1 - probability_home

        return {
            "predicted_mov": predicted_mov,
            "probability_home": probability_home,
            "probability_away": probability_away,
            "fair_odds_home": 1 / probability_home,
            "fair_odds_away": 1 / probability_away,
        }

    def calculate_bet(
        self, probability: float, broker_odds: float, bankroll: float = 1000
    ) -> Dict:
        """
        Calculate betting metrics for a given probability and odds.

        Args:
            probability (float): Probability of winning
            broker_odds (float): Odds offered by broker
            bankroll (float, optional): Available bankroll. Defaults to 1000.

        Returns:
            Dict: Dictionary containing betting metrics
        """
        kelly_criterion = ((broker_odds - 1) * probability - (1 - probability)) / (
            broker_odds - 1
        )
        expected_value = broker_odds * probability - 1
        bet_size = bankroll * kelly_criterion

        return {
            "kelly_criterion": kelly_criterion,
            "expected_value": expected_value,
            "bet_size": bet_size,
        }

    def _log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """
        Calculate negative log likelihood for parameter optimization.

        Args:
            params (np.ndarray): Model parameters
            data (np.ndarray): Training data

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
            result = logistic if data[i][2] == 1 else 1 - logistic
            log_likelihood += np.log(result)

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


if __name__ == "__main__":
    # Initialize and fit the model
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(league="Herre Handbold Ligaen")

    model = BradleyTerryModel()
    model.fit(df)

    # Make predictions
    home_team = "Kolding"
    away_team = "Bjerringbro/Silkeborg"
    prediction = model.predict(home_team, away_team)

    # Calculate betting metrics
    home_bet = model.calculate_bet(prediction["probability_home"], broker_odds=6.50)
    away_bet = model.calculate_bet(prediction["probability_away"], broker_odds=1.22)
