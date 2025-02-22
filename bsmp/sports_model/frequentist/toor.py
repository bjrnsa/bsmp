# %%
from pathlib import Path

import scipy.stats as stats
import statsmodels.api as sm
from util import logit_strength_ratings

data_path = Path().cwd().parent.parent / "data/raw/afl_data.csv"

from bsmp.data_models.data_loader import MatchDataLoader

loader = MatchDataLoader(sport="handball")
df = loader.load_matches(
    league="Herre Handbold Ligaen",
    # seasons=["2023/2024"],
)
# %%
# Create a dictionary of team with a rating of 1 to start
teams = df["home_team"].unique()
data = df.loc[:176, ["home_team", "away_team", "result"]].values
# Map results back to team_ratings and HFA
insample_team_ratings, hfa_estimate = logit_strength_ratings(data, teams)

# Add to df
df["home_rating"] = df["home_team"].map(insample_team_ratings)
df["away_rating"] = df["away_team"].map(insample_team_ratings)


# %% Add "SSE Min Function" column to df
# Initialize three moor parameters in a dict
ols_data = df.loc[:176, ["home_rating", "away_rating", "home_mov"]]
X = sm.add_constant(ols_data[["home_rating", "away_rating"]])
y = ols_data["home_mov"]
ols = sm.OLS(y, X).fit()
ols.summary()


df["sse_min"] = ols.predict(sm.add_constant(df[["home_rating", "away_rating"]]))
df["mov_error_sqr"] = (df["sse_min"] - df["home_mov"]) ** 2

df["mov_error_sqr"].sum()

# Redefine insample data
insample_df = df.iloc[:177]

model = sm.OLS(insample_df["home_mov"], sm.add_constant(insample_df["sse_min"]))
reg = model.fit()
reg.summary()

intercept = reg.params["const"]
slope = reg.params["sse_min"]
std_error_resid = reg.resid.std()

# %% Predict
df["pred_mov"] = intercept + slope * df["sse_min"]

AWAY_TEAM = "North Melbourne"
HOME_TEAM = "St Kilda"

sse_val = (
    intercept
    + slope
    * df.query("home_team == @HOME_TEAM & away_team == @AWAY_TEAM")["sse_min"].values[0]
)

predicted_mov = intercept + slope * (1 - sse_val)
point_spread_vic = 0

probability_vic = 1 - stats.norm.cdf(point_spread_vic, predicted_mov, std_error_resid)
fair_odds = 1 / probability_vic
broker_odds = 1.35
kelly_criterion = ((broker_odds - 1) * probability_vic - (1 - probability_vic)) / (
    broker_odds - 1
)
ev = broker_odds * probability_vic - 1

print(f"SSE Value: {sse_val}")
print(f"Predicted MOV: {predicted_mov}")
print(f"Probability of Victory: {probability_vic}")
print(f"Fair Odds: {fair_odds}")
print(f"Broker Odds: {broker_odds}")
print(f"Kelly Criterion: {kelly_criterion}")
print(f"EV: {ev}")
# %%
