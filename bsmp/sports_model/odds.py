# %%
"""Module for sports betting odds analysis and model evaluation using various prediction methods."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.ensemble import NNLS, SimpleEnsemble
from bsmp.sports_model.evaluator import OddsEvaluator
from bsmp.sports_model.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry, Poisson
from bsmp.sports_model.implied_odds import ImpliedOdds
from bsmp.sports_model.utils import dixon_coles_weights

loader = MatchDataLoader(sport="handball")
df = loader.load_matches(
    league="Herre Handbold Ligaen",
    seasons=["2024/2025"],
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
bookmaker_odds = (
    loader.load_odds(test_df.index)
    .groupby("flashscore_id")
    .max()
    .filter(regex="odds")
    .join(test_df, how="right")
    .sort_values("datetime")
    .filter(regex="odds")
)
# bookmaker_odds contains the best odds for each match indexed by match id
# and have three columns home_win_odds, draw_odds and away_win_odds
weights = dixon_coles_weights(train_df.datetime, xi=0.018)
implied = ImpliedOdds()
implied_probas = implied.get_implied_probabilities(bookmaker_odds)
implied_probas = implied_probas.pivot_table(
    index="match_id", columns="outcome", values="power"
)
margins = implied.get_margins(bookmaker_odds)

model_list = [
    "bradley_terry",
    "gssd",
    "prp",
    "toor",
    "zsd",
    "dixon_coles",
    "simple_ensemble",
    "nnls_ensemble",
]

# Create and fit the model
models = [
    BradleyTerry(),
    GSSD(),
    PRP(),
    TOOR(),
    ZSD(),
    Poisson(),
    SimpleEnsemble(model_names=["zsd", "gssd", "prp"]),
    NNLS(model_names=["zsd", "gssd", "prp"]),
]
predictions = pd.DataFrame(0.0, columns=model_list, index=test_df.index)
draw_probas = pd.DataFrame(0.0, columns=model_list, index=test_df.index)
home_probas = pd.DataFrame(0.0, columns=model_list, index=test_df.index)
away_probas = pd.DataFrame(0.0, columns=model_list, index=test_df.index)
outcomes = ["home_win", "away_win", "draw"]

for model_name, model in zip(model_list, models):
    model.fit(
        y=train_df["goal_difference"],
        X=train_df[["home_team", "away_team"]],
        Z=train_df[["home_goals", "away_goals"]],
        weights=weights,
    )
    preds = model.predict(X=test_df[["home_team", "away_team"]])
    predictions.loc[test_df.index, model_name] = preds

    for outcome in outcomes:
        probas = model.predict_proba(
            X=test_df[["home_team", "away_team"]], outcome=outcome
        )
        if outcome == "draw":
            draw_probas.loc[test_df.index, model_name] = probas
        elif outcome == "home_win":
            home_probas.loc[test_df.index, model_name] = probas
        elif outcome == "away_win":
            away_probas.loc[test_df.index, model_name] = probas

# Create an instance of the evaluator
evaluator = OddsEvaluator(kelly_fraction=0.2, market_efficiency=0.7)

# Create a dictionary of model probabilities
model_probas = {"home": home_probas, "draw": draw_probas, "away": away_probas}

# Evaluate all betting opportunities
# Evaluate opportunities with Benter boost
opportunities = evaluator.evaluate_opportunities(
    bookmaker_odds=bookmaker_odds,
    model_probas=model_probas,
    model_names=model_list,
    # implied_probas=implied_probas,  # Pass your implied probabilities DataFrame
)
# Get the best betting opportunities
best_bets = evaluator.get_best_bets(
    opportunities=opportunities,
    min_ev=2.0,
    # min_kelly=0.05,
)

for model, model_df in best_bets.groupby("model"):
    print(model)

    model_df["bet_size"] = model_df["kelly"] * 1000
    model_df["actual_outcome"] = test_df.loc[model_df.index, "result"]
    model_df["bet_outcome"] = model_df["actual_outcome"] * model_df["outcome"]
    model_df["pnl"] = model_df.apply(
        lambda x: x["bet_size"] * x["bookmaker_odds"]
        if x["bet_outcome"] == 1
        else -x["bet_size"],
        axis=1,
    )

    print(model_df["pnl"].sum())

    model_df["pnl"].cumsum().plot()
    plt.title(str(model))
    plt.show()


# %%
