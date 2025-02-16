# %%
import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
from cmdstanpy import CmdStanModel

from bsmp.data_models.data_loader import MatchDataLoader

stan_file = "bsmp/sports_model/stan_files/test_model.stan"
loader = MatchDataLoader(sport="handball")
df = loader.load_matches(
    season="2024/2025",
    league="Herre Handbold Ligaen",
    date_range=("2024-08-01", "2025-02-01"),
    team_filters={"country": "Denmark"},
)

data = {
    "N": len(df),
    "T": len(loader.team_list),
    "home_team": df["home_index"].values,
    "away_team": df["away_index"].values,
    "home_goals": df["hg"].values,
    "away_goals": df["ag"].values,
}

model = CmdStanModel(stan_file=stan_file)
draws = 8000
warmup = 1000
chains = 6
seed = 123
max_tree_depth = 10

result = model.sample(
    data=data,
    iter_sampling=draws,
    iter_warmup=warmup,
    chains=chains,
    seed=seed,
    max_treedepth=max_tree_depth,
)

# Calculate total number of simulations
n_samples = result.draws_pd().shape[0]


result_draws = result.draws_pd()
result_draws.filter(regex="119")

# Define coordinate system for teams
coords = {
    "team": loader.team_list,  # Your actual team names
    "match": np.arange(len(df)),
}

# Map dimensions to parameters
dims = {
    "att": ["team"],
    "def": ["team"],
    "home_team": ["match"],
    "away_team": ["match"],
    "home_goals": ["match"],
    "away_goals": ["match"],
}

observed_data = {
    "home_goals": df["hg"].values,
    "away_goals": df["ag"].values,
}
constant_data = {
    "home_team": df["home_index"].values,
    "away_team": df["away_index"].values,
}

posterior_predictive = [
    "lambda_home",
    "lambda_away",
    "home_goals_sim",
    "away_goals_sim",
]

# Convert to InferenceData
idata = az.from_cmdstanpy(
    posterior=result,
    observed_data=observed_data,
    constant_data=constant_data,
    coords=coords,
    dims=dims,
    posterior_predictive=posterior_predictive,
)

az.plot_trace(idata, var_names=["att", "def", "home_advantage", "goal_mean"])
az.plot_posterior(idata, var_names=["att", "def", "home_advantage", "goal_mean"])

idata.posterior["log_lambda_home"].stack(sample=("chain", "draw")).to_dataframe().head()


def plot_team_strength(idata):
    """Plot posterior density of team strengths using Plotly.

    Args:
        idata (arviz.InferenceData): Result from az.from_cmdstanpy()
        team_names (list): Ordered list of team names matching model indices
    """
    # Extract posterior samples and calculate aggregate strength
    post = idata.posterior
    agg_strength = post["att"] - post["def"]

    # Convert to long format DataFrame
    df_long = (
        agg_strength.stack(sample=("chain", "draw"))
        .to_dataframe(name="Strength")
        .reset_index(["team"])
    )

    # Create density plot
    fig = px.histogram(
        df_long,
        x="Strength",
        facet_col="team",
        facet_col_wrap=5,
        color="team",
        height=900,
        width=1200,
        title="Posterior Team Strength Distributions",
    )

    # Add vertical line and styling
    fig.update_layout(
        showlegend=False,
        xaxis_title="Aggregate Strength (att - def)",
        yaxis_title="Density",
        template="ggplot2",  # Use ggplot-style template
    )

    fig.add_vline(x=0, line_dash="dot", line_color="red")

    # Adjust facet labels
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.show()


plot_team_strength(idata)


def simulate_match(post, home_team, away_team, n_samples=1000):
    """Simulate match outcomes based on posterior samples.

    Args:
        post (arviz.InferenceData): Posterior samples from model
        home_team (str): Home team name
        away_team (str): Away team name
        n_samples (int): Number of samples to draw

    Returns:
        pd.DataFrame: DataFrame with simulated match outcomes
    """
    # Extract posterior samples
    att = post.posterior["att"].to_dataframe().reset_index(["team"])
    def_ = post.posterior["def"].to_dataframe().reset_index(["team"])
    home_adv = post.posterior["home_advantage"].to_dataframe()
    goal_mean = post.posterior["goal_mean"].to_dataframe()

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

    return results


home_team = "Aalborg"
away_team = "Bjerringbro/Silkeborg"
match_results = simulate_match(idata, home_team, away_team)


def calculate_metrics(match_results):
    """Calculate match outcome probabilities and metrics."""

    home_win_prob = np.mean(match_results["home_goals"] > match_results["away_goals"])
    away_win_prob = np.mean(match_results["away_goals"] > match_results["home_goals"])
    draw_prob = np.mean(match_results["home_goals"] == match_results["away_goals"])
    home_win_odd = 1 / home_win_prob
    away_win_odd = 1 / away_win_prob
    draw_odd = 1 / draw_prob
    home_goals = np.mean(match_results["home_goals"])
    away_goals = np.mean(match_results["away_goals"])
    total_goals = np.mean(match_results["home_goals"] + match_results["away_goals"])

    metrics = pd.Series(
        {
            "Home Win Probability": home_win_prob,
            "Away Win Probability": away_win_prob,
            "Draw Probability": draw_prob,
            "Home Goals": home_goals,
            "Away Goals": away_goals,
            "Total Goals": total_goals,
            "Home Win Odds": home_win_odd,
            "Away Win Odds": away_win_odd,
            "Draw Odds": draw_odd,
        }
    )

    return metrics.round(3)


match_metrics = calculate_metrics(match_results)

# %%
