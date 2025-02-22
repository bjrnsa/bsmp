# %%
import numpy as np
from scipy import stats

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerryModel
from bsmp.sports_model.utils import dixon_coles_weights


class TOOR(BradleyTerryModel):
    def __init__(self, df, weights, hfa=0.1):
        super().__init__(df, weights, hfa)

    def fit(self):
        self.best_params = self.bradley_terry_params()
        self.team_ratings = dict(zip(self.teams, self.best_params[:-1]))
        home_ratings = np.array([self.team_ratings[team] for team in self.home_team])
        away_ratings = np.array([self.team_ratings[team] for team in self.away_team])
        goal_difference = self.goal_difference
        X = np.column_stack((home_ratings, away_ratings))
        self.team_betas, _ = self._ols(goal_difference, X)
        fitted_values = np.dot(np.column_stack((np.ones(len(X)), X)), self.team_betas)
        (self.const, self.beta), self.std_error = self._ols(
            goal_difference, fitted_values
        )
        self.team_reg_ratings = dict(zip(fitted_values, self.team_ratings))

    def predict(self, home_team, away_team, point_spread=0.0):
        logit_val = self._logit_strength(home_team, away_team)
        pred = self.const + self.beta * logit_val

        prob_home = 1 - stats.norm.cdf(point_spread + 0.5, pred, self.std_error)
        prob_away = stats.norm.cdf(-point_spread - 0.5, pred, self.std_error)
        prob_draw = 1 - prob_home - prob_away

        return prob_home, prob_draw, prob_away

    def _reg_strength(self, data, teams):
        self.team_betas


if __name__ == "__main__":
    # Initialize and fit the model
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(league="Herre Handbold Ligaen")
    odds_df = loader.load_odds(df.index.unique())
    weights = dixon_coles_weights(df.datetime, xi=0.18)

    model = TOOR(df, weights, 0.1)
    model.fit()

    # Make predictions
    home_team = "Kolding"
    away_team = "Bjerringbro/Silkeborg"
    prob_home, prob_draw, prob_away = model.predict(home_team, away_team)
