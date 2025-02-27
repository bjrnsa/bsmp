import numpy as np
from numba import njit


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
