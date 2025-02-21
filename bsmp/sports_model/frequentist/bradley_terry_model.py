# %%
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import approx_fprime, minimize

from bsmp.data_models.data_loader import MatchDataLoader

loader = MatchDataLoader(sport="handball")
df = loader.load_matches(
    league="Herre Handbold Ligaen",
    # seasons=["2023/2024"],
)


# %%
# Create a dictionary of team with a rating of 1 to start
teams = df["home_team"].unique()
team_ratings = dict.fromkeys(teams, 1.0)

# Initial home field advantage value used in excel
HFA = 0.1

# Create params dict with initial ratings and HFA
params = np.array(list(team_ratings.values()) + [HFA])


# Create log likelihood function that is numpy-only
def log_likelihood_np(params, teams, data):
    ratings = params[:-1]
    HFA = params[-1]

    log_likelihood = 0.0

    for i, team in enumerate(data):
        HT = ratings[teams == team[0]][0]
        AT = ratings[teams == team[1]][0]

        logistic = 1 / (1 + np.exp(-(HFA + HT - AT)))
        result = logistic if data[i][2] == 1 else 1 - logistic
        log_likelihood += np.log(result)

    return -log_likelihood


# Set insample and outsample data
insample_data = df.query("season_year != 2025")[
    ["home_team", "away_team", "result"]
].values
nobs = len(insample_data)


def func(params):
    return log_likelihood_np(params, teams, insample_data) / nobs


def grad(params):
    return approx_fprime(params, func, 6.5e-07) / nobs


result = minimize(
    func, params, jac=grad, method="SLSQP", options={"ftol": 1e-10, "maxiter": 200}
)
best_params = result.x

# Map results back to team_ratings and HFA
insample_team_ratings = dict(zip(teams, best_params[:-1]))
hfa_estimate = best_params[-1]

# Add to df
df["home_rating"] = df["home_team"].map(insample_team_ratings)
df["away_rating"] = df["away_team"].map(insample_team_ratings)


# Define the logistic rating transform function
def logit_transform(x):
    return 1 / (1 + np.exp(-x))


df["logit_scale"] = df.apply(
    lambda x: logit_transform(hfa_estimate + x["home_rating"] - x["away_rating"]),
    axis=1,
)

# Redefine insample data
oos_df = df.query("season_year == 2025")

# Run a regression where y is Home MOV and x is the logistic function
model = sm.OLS(oos_df["goal_difference"], sm.add_constant(oos_df["logit_scale"]))
reg = model.fit()
reg.summary()

intercept = reg.params["const"]
slope = reg.params["logit_scale"]
std_error_resid = reg.resid.std()


# %% Predict
df["pred_mov"] = intercept + slope * df["logit_scale"]

HOME_TEAM = "Kolding"
AWAY_TEAM = "Bjerringbro/Silkeborg"

logit_val = logit_transform(
    -(
        hfa_estimate
        + insample_team_ratings[HOME_TEAM]
        - insample_team_ratings[AWAY_TEAM]
    )
)


def logit_strength(ratings, hfa, home_team, away_team):
    logit_val = logit_transform(-(hfa + ratings[home_team] - ratings[away_team]))
    return logit_val


predicted_mov = intercept + slope * (1 - logit_val)
point_spread_vic = 1

probability_home = 1 - stats.norm.cdf(point_spread_vic, predicted_mov, std_error_resid)
fair_odds_home = 1 / probability_home
broker_odds_home = 6.50
kelly_criterion_home = (
    (broker_odds_home - 1) * probability_home - (1 - probability_home)
) / (broker_odds_home - 1)
ev_home = broker_odds_home * probability_home - 1
bankroll = 1000
bet_size_home = bankroll * kelly_criterion_home

probability_away = 1 - probability_home
fair_odds_away = 1 / probability_away
broker_odds_away = 1.22
kelly_criterion_away = (
    (broker_odds_away - 1) * probability_away - (1 - probability_away)
) / (broker_odds_away - 1)
ev_away = broker_odds_away * probability_away - 1

bankroll = 1000
bet_size_away = bankroll * kelly_criterion_away

print(
    f"Home team: {HOME_TEAM} \nAway team: {AWAY_TEAM} \n\nPredicted MOV: {predicted_mov:.2f}\n\n"
    f"Probability of home team winning: {probability_home:.2f}\n"
    f"Kelly Criterion for home team: {kelly_criterion_home:.2f}\n"
    f"EV for home team: {ev_home:.2f}\n\n"
    f"Bet size for home team: {bet_size_home:.2f}\n\n"
    f"Probability of away team winning: {probability_away:.2f}\n"
    f"Kelly Criterion for away team: {kelly_criterion_away:.2f}\n"
    f"EV for away team: {ev_away:.2f}\n\n"
    f"Bet size for away team: {bet_size_away:.2f}\n\n"
)
# %%
