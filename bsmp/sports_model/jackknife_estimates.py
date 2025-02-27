# %%
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.frequentist.bradley_terry_model import BradleyTerry
from bsmp.sports_model.metrics.metrics import Metrics
from bsmp.sports_model.utils import dixon_coles_weights


@njit
def log_likelihood(
    params: np.ndarray,
    ratings: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    result: np.ndarray,
    ratings_weights: np.ndarray,
) -> float:
    """Calculate negative log likelihood for parameter optimization."""
    ratings = params[:-1]
    home_advantage = params[-1]
    log_likelihood = 0.0

    # Precompute home and away ratings
    home_ratings = ratings[home_idx]
    away_ratings = ratings[away_idx]
    win_probs = 1 / (1 + np.exp(-(home_advantage + home_ratings - away_ratings)))

    # Vectorized calculation
    win_mask = result == 1
    loss_mask = result == -1
    draw_mask = ~(win_mask | loss_mask)

    log_likelihood += np.sum(ratings_weights[win_mask] * np.log(win_probs[win_mask]))
    log_likelihood += np.sum(
        ratings_weights[loss_mask] * np.log(1 - win_probs[loss_mask])
    )
    log_likelihood += np.sum(
        ratings_weights[draw_mask]
        * (np.log(win_probs[draw_mask]) + np.log(1 - win_probs[draw_mask]))
    )

    return -log_likelihood


class BradleyTerryNumba(BradleyTerry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP."""
        result, _ = jackknife_estimates(
            params=self.params_,
            result=self.result_,
            ratings_weights=self.weights_,
            home_idx=self.home_idx_,
            away_idx=self.away_idx_,
            log_likelihood_func=log_likelihood,
        )

        return result


def jackknife_estimates(
    params: np.ndarray,
    result: np.ndarray,
    ratings_weights: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    log_likelihood_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate jackknife estimates of the model parameters.

    Args:
        params (np.ndarray): Initial model parameters.
        result (np.ndarray): Array of match results.
        ratings_weights (np.ndarray): Weights for the matches.
        home_idx (np.ndarray): Indices of home teams.
        away_idx (np.ndarray): Indices of away teams.
        log_likelihood_func: Function to calculate log likelihood.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - jackknife estimates of the parameters
            - standard errors of the jackknife estimates
    """
    n = len(result)
    jackknife_params = np.zeros((n, len(params)))

    for i in range(n):
        # Leave out the i-th observation
        reduced_result = np.delete(result, i)
        reduced_weights = np.delete(ratings_weights, i)

        # Optimize parameters again
        jackknife_result = minimize(
            fun=lambda p: log_likelihood(
                p, params[:-1], home_idx, away_idx, reduced_result, reduced_weights
            )
            / len(reduced_result),
            x0=params,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )

        # Store the parameters
        jackknife_params[i] = jackknife_result.x

    # Calculate jackknife estimates
    jackknife_estimates = np.mean(jackknife_params, axis=0)

    # Calculate jackknife standard errors
    jackknife_se = np.sqrt(
        (n - 1) * np.mean((jackknife_params - jackknife_estimates) ** 2, axis=0)
    )

    return jackknife_estimates, jackknife_se


# %%
if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Herre Handbold Ligaen",
        seasons=["2024/2025"],
    )
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
    weights = dixon_coles_weights(train_df.datetime, xi=0.0018)

    home_team = "Kolding"
    away_team = "Sonderjyske"

    # Create and fit the model
    model = BradleyTerryNumba()
    model_complete = BradleyTerry()

    # Prepare training data
    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    Z_train = train_df[["home_goals", "away_goals"]]
    model.fit(X_train, y_train, Z=Z_train, weights=weights)
    model_complete.fit(X_train, y_train, Z=Z_train, weights=weights)
    # Display team ratings
    print(model.get_team_ratings())
    print(model_complete.get_team_ratings())
    # Create a test DataFrame for prediction
    X_test = test_df[["home_team", "away_team"]]

    # Predict point spreads (goal differences)
    predicted_spreads = model.predict(X_test)
    predicted_spreads_complete = model_complete.predict(X_test)
    print(f"Predicted goal difference: {predicted_spreads[0]:.2f}")
    print(f"Predicted goal difference: {predicted_spreads_complete[0]:.2f}")

    # Predict probabilities
    probs = model.predict_proba(
        X_test, point_spread=0, include_draw=True, outcome="draw"
    )
    probs_complete = model_complete.predict_proba(
        X_test, point_spread=0, include_draw=True, outcome="draw"
    )

    y_test = test_df["goal_difference"]

    preds = pd.DataFrame(0.0, columns=["numba", "complete"], index=test_df.index)
    preds.loc[test_df.index, "numba"] = predicted_spreads
    preds.loc[test_df.index, "complete"] = predicted_spreads_complete

    probs_df = pd.DataFrame(0.0, columns=["numba", "complete"], index=test_df.index)
    probs_df.loc[test_df.index, "numba"] = probs
    probs_df.loc[test_df.index, "complete"] = probs_complete

    metrics = Metrics(
        predictions=preds,
        probabilities=probs_df,
        true_values=y_test,
    )
    print(metrics.regression_metrics_to_df())
    print(metrics.classification_metrics_to_df(outcome="draw"))


# %%
