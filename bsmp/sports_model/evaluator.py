"""This module contains the OddsEvaluator class for evaluating betting opportunities using Kelly criterion and Expected Value."""

import numpy as np
import pandas as pd


class OddsEvaluator:
    """Class to evaluate betting opportunities using Kelly criterion and Expected Value."""

    def __init__(self, kelly_fraction: float = 0.5, market_efficiency: float = 0.8):
        """Initialize the OddsEvaluator.

        Args:
            kelly_fraction (float): Fraction of Kelly criterion to use (default: 0.5 for half Kelly)
            market_efficiency (float): Weight given to market probabilities in Benter boost (default: 0.8)
        """
        self.kelly_fraction = kelly_fraction
        self.market_efficiency = market_efficiency

    def calculate_ev(self, probability: float, odds: float) -> float:
        """Calculate Expected Value for a bet.

        Args:
            probability (float): Model's predicted probability
            odds (float): Bookmaker odds

        Returns:
            float: Expected value as a percentage
        """
        return (probability * (odds - 1) - (1 - probability)) * 100

    def calculate_kelly(self, probability: float, odds: float) -> float:
        """Calculate Kelly criterion stake.

        Args:
            probability (float): Model's predicted probability
            odds (float): Bookmaker odds

        Returns:
            float: Recommended stake as a fraction of bankroll
        """
        if odds <= 1 or probability <= 0:
            return 0

        q = 1 - probability
        kelly = (probability * (odds - 1) - q) / (odds - 1)
        kelly = max(0, kelly * self.kelly_fraction)
        return kelly

    def apply_benter_boost(
        self,
        model_probs: pd.Series,
        implied_probs: pd.Series,
    ) -> pd.Series:
        """Apply Benter boost to combine model and market probabilities.

        Args:
            model_probs: Series of model probabilities
            implied_probs: Series of implied market probabilities

        Returns:
            Series of boosted probabilities
        """
        # Convert to log odds
        model_log_odds = np.log(model_probs)
        market_log_odds = np.log(implied_probs)

        model_log_weights = (1 - self.market_efficiency) * model_log_odds
        market_log_weights = self.market_efficiency * market_log_odds

        boosted_probs = np.exp(model_log_weights + market_log_weights)
        total_boosted = boosted_probs.sum()

        return boosted_probs / total_boosted

    def evaluate_opportunities(
        self,
        bookmaker_odds: pd.DataFrame,
        model_probas: dict[str, pd.DataFrame],
        model_names: list[str],
        implied_probas: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Evaluate betting opportunities across all outcomes and models.

        Args:
            bookmaker_odds (pd.DataFrame): DataFrame with columns [home_win_odds, draw_odds, away_win_odds]
            model_probas (dict): Dictionary with keys 'home', 'draw', 'away' containing DataFrames of model probabilities
            model_names (list): List of model names to evaluate
            implied_probas (pd.DataFrame, optional): DataFrame with implied probabilities from bookmaker odds

        Returns:
            pd.DataFrame: DataFrame with EV and Kelly values for each bet opportunity
        """
        outcomes = ["home_win", "draw", "away_win"]
        odds_cols = ["home_odds", "draw_odds", "away_odds"]

        results = []

        for model in model_names:
            for outcome, odds_col in zip(outcomes, odds_cols):
                model_probas
                if outcome.startswith("home"):
                    probas_1 = model_probas["home"][model]
                    probas_2 = model_probas["draw"][model]
                    probas_3 = model_probas["away"][model]
                    probas = model_probas["home"][model]

                    outcome_int = 1
                    implied_key = "home_win"
                elif outcome.startswith("away"):
                    probas_1 = model_probas["home"][model]
                    probas_2 = model_probas["draw"][model]
                    probas_3 = model_probas["away"][model]
                    probas = model_probas["away"][model]
                    outcome_int = -1
                    implied_key = "away_win"
                else:
                    probas_1 = model_probas["home"][model]
                    probas_2 = model_probas["draw"][model]
                    probas_3 = model_probas["away"][model]
                    probas = model_probas["draw"][model]
                    outcome_int = 0
                    implied_key = "draw"

                odds = bookmaker_odds[odds_col]

                for idx in odds.index:
                    prob = probas.loc[idx]
                    odd = odds.loc[idx]
                    probs = pd.DataFrame(
                        {
                            "home_win": probas_1,
                            "draw": probas_2,
                            "away_win": probas_3,
                        }
                    ).loc[idx]

                    # Apply Benter boost if implied probabilities are provided
                    if implied_probas is not None:
                        implied_prob = implied_probas.loc[idx]
                        prob = self.apply_benter_boost(
                            probs,
                            implied_prob,
                        )
                        prob = prob.loc[outcome]

                    ev = self.calculate_ev(prob, odd)
                    kelly = self.calculate_kelly(prob, odd)

                    results.append(
                        {
                            "match_id": idx,
                            "model": model,
                            "outcome": outcome_int,
                            "model_probability": prob,
                            "model_odds": 1 / prob,
                            "bookmaker_odds": odd,
                            "ev": ev,
                            "kelly": kelly,
                        }
                    )

        return pd.DataFrame(results)

    def get_best_bets(
        self, opportunities: pd.DataFrame, min_ev: float = 2.0, min_kelly: float = 0.01
    ) -> pd.DataFrame:
        """Filter for the best betting opportunities.

        Args:
            opportunities (pd.DataFrame): DataFrame from evaluate_opportunities
            min_ev (float): Minimum expected value percentage
            min_kelly (float): Minimum Kelly criterion value

        Returns:
            pd.DataFrame: Filtered DataFrame with the best betting opportunities
        """
        opp_df = opportunities[
            (opportunities["ev"] >= min_ev) & (opportunities["kelly"] >= min_kelly)
        ].set_index("match_id")

        return opp_df
