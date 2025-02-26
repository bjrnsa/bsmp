# %%
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.ensemble import Ensemble
from bsmp.sports_model.utils import dixon_coles_weights


class OddsEvaluator:
    """
    A class for evaluating betting opportunities by comparing bookmaker odds with model probabilities.

    This class calculates:
    - Expected Value (EV): model_probability * bookmaker_odds - 1
    - Kelly Criterion: optimal bet size based on edge and odds
    """

    def __init__(self, fraction: float = 1.0):
        """
        Initialize the OddsEvaluator.

        Parameters:
        -----------
        fraction : float, default=1.0
            Kelly fraction to use (between 0 and 1).
            - 1.0 = full Kelly (theoretically optimal but high variance)
            - Lower values (e.g., 0.25) = more conservative betting
        """
        self.fraction = fraction

    def _calculate_kelly(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly Criterion bet size.

        Parameters:
        -----------
        prob : float
            Estimated probability of winning (0 to 1)
        odds : float
            Decimal odds offered by bookmaker

        Returns:
        --------
        float
            Kelly stake as a fraction of bankroll
        """
        # Kelly formula: (bp - q) / b
        # where b = odds - 1, p = probability of winning, q = probability of losing
        b = odds - 1  # Potential profit on a 1 unit stake
        q = 1 - prob  # Probability of losing

        if prob * odds > 1:  # Only bet when there's positive expected value
            kelly = (prob * b - q) / b
            return max(0, kelly)  # Apply fraction and ensure non-negative
        else:
            return 0.0

    def _calculate_ev(self, prob: float, odds: float) -> float:
        """
        Calculate Expected Value.

        Parameters:
        -----------
        prob : float
            Estimated probability of winning (0 to 1)
        odds : float
            Decimal odds offered by bookmaker

        Returns:
        --------
        float
            Expected value as a percentage
        """
        return prob * odds - 1

    def evaluate_opportunities(
        self, bookmaker_odds: pd.DataFrame, model_probabilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate betting opportunities by comparing bookmaker odds with model probabilities.

        Parameters:
        -----------
        bookmaker_odds : pd.DataFrame
            DataFrame with bookmaker odds, indexed by flashscore_id with columns:
            - bookmaker: name of the bookmaker
            - home_odds: decimal odds for home win
            - draw_odds: decimal odds for draw
            - away_odds: decimal odds for away win

        model_probabilities : pd.DataFrame
            DataFrame with model probabilities, indexed by flashscore_id with columns:
            - home_prob: probability of home win
            - draw_prob: probability of draw
            - away_prob: probability of away win

        Returns:
        --------
        pd.DataFrame
            DataFrame with evaluation results including EV and Kelly criterion
        """
        results = []

        # Ensure both DataFrames have the same index
        common_matches = set(bookmaker_odds.index).intersection(
            set(model_probabilities.index)
        )

        for flashscore_id in common_matches:
            # Get model probabilities for this match
            if flashscore_id not in model_probabilities.index:
                continue

            match_probs = model_probabilities.loc[flashscore_id]

            # Get all bookmaker odds for this match
            match_odds_rows = bookmaker_odds.loc[flashscore_id]

            # Handle case where there's only one bookmaker (convert to DataFrame)
            if not isinstance(match_odds_rows, pd.DataFrame):
                match_odds_rows = pd.DataFrame([match_odds_rows])

            for _, odds_row in match_odds_rows.iterrows():
                bookmaker = odds_row["bookmaker"]

                # Calculate EV and Kelly for each outcome
                outcomes = [
                    ("home", match_probs["home_prob"], odds_row["home_odds"], 1),
                    ("draw", match_probs["draw_prob"], odds_row["draw_odds"], 0),
                    ("away", match_probs["away_prob"], odds_row["away_odds"], -1),
                ]

                for outcome, prob, odds, outcome_in_model in outcomes:
                    ev = self._calculate_ev(prob, odds)
                    kelly = self._calculate_kelly(prob, odds)

                    results.append(
                        {
                            "flashscore_id": flashscore_id,
                            "bookmaker": bookmaker,
                            "bet_outcome": outcome,
                            "outcome_in_model": outcome_in_model,
                            "model_probability": prob,
                            "bookmaker_odds": odds,
                            "ev": ev,
                            "kelly": kelly,
                            "fractional_kelly": kelly * self.fraction,
                            "ev_percentage": ev * 100,  # EV as percentage
                        }
                    )

        return pd.DataFrame(results)

    def get_best_opportunities(
        self,
        evaluated_df: pd.DataFrame,
        min_ev: float = 0.0,
        min_kelly: float = 0.0,
        min_odds: float = 1.0,
        max_odds: float = float("inf"),
        bookmakers: Optional[List[str]] = None,
        selection_method: str = "max_ev",
    ) -> pd.DataFrame:
        """
        Filter for the best betting opportunities based on criteria.

        Parameters:
        -----------
        evaluated_df : pd.DataFrame
            DataFrame from evaluate_opportunities method
        min_ev : float, default=0.0
            Minimum expected value to include
        min_kelly : float, default=0.0
            Minimum Kelly criterion to include
        min_odds : float, default=1.0
            Minimum odds to include
        max_odds : float, default=inf
            Maximum odds to include
        bookmakers : List[str], optional
            List of bookmakers to include (if None, include all)
        selection_method : str, default="max_ev"
            Method to select best opportunity per match:
            - "max_ev": Select opportunity with highest EV
            - "max_kelly": Select opportunity with highest Kelly criterion
            - "max_odds": Select opportunity with highest odds
            - "all": Return all opportunities that meet criteria

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame with best opportunities
        """
        filtered = evaluated_df.copy()

        # Apply filters
        filtered = filtered[filtered["ev"] >= min_ev]
        filtered = filtered[filtered["kelly"] >= min_kelly]
        filtered = filtered[filtered["bookmaker_odds"] >= min_odds]
        filtered = filtered[filtered["bookmaker_odds"] <= max_odds]

        if bookmakers is not None:
            filtered = filtered[filtered["bookmaker"].isin(bookmakers)]

        # Select best opportunity per match based on selection method
        if selection_method != "all" and not filtered.empty:
            if selection_method == "max_ev":
                filtered = filtered.loc[
                    filtered.groupby("flashscore_id")["ev"].idxmax()
                ]
            elif selection_method == "max_kelly":
                filtered = filtered.loc[
                    filtered.groupby("flashscore_id")["kelly"].idxmax()
                ]
            elif selection_method == "max_odds":
                filtered = filtered.loc[
                    filtered.groupby("flashscore_id")["bookmaker_odds"].idxmax()
                ]

        return filtered

    def get_best_odds(
        self, bookmaker_odds: pd.DataFrame, model_probabilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each match and outcome, find the bookmaker offering the best odds.

        Parameters:
        -----------
        bookmaker_odds : pd.DataFrame
            DataFrame with bookmaker odds
        model_probabilities : pd.DataFrame
            DataFrame with model probabilities

        Returns:
        --------
        pd.DataFrame
            DataFrame with best odds for each match and outcome
        """
        # First evaluate all opportunities
        all_opportunities = self.evaluate_opportunities(
            bookmaker_odds, model_probabilities
        )

        # Group by flashscore_id and bet_outcome, then find the row with the highest odds
        best_odds = all_opportunities.loc[
            all_opportunities.groupby(["flashscore_id", "bet_outcome"])[
                "bookmaker_odds"
            ].idxmax()
        ]

        return best_odds

    def summarize_by_bookmaker(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize opportunities by bookmaker.

        Parameters:
        -----------
        evaluated_df : pd.DataFrame
            DataFrame from evaluate_opportunities method

        Returns:
        --------
        pd.DataFrame
            Summary statistics by bookmaker
        """
        return (
            evaluated_df.groupby("bookmaker")
            .agg(
                {
                    "ev": ["mean", "min", "max", "count"],
                    "kelly": ["mean", "min", "max"],
                    "bookmaker_odds": ["mean", "min", "max"],
                }
            )
            .sort_values(("ev", "mean"), ascending=False)
        )

    def run_backtest(
        self,
        opportunities: pd.DataFrame,
        results_df: pd.DataFrame,
        initial_bankroll: float = 1000.0,
        stake_method: str = "fractional_kelly",
        max_stake_pct: float = 0.1,
        min_bankroll_pct: float = 0.1,
    ) -> pd.DataFrame:
        """
        Run a backtest on betting opportunities.

        Parameters:
        -----------
        opportunities : pd.DataFrame
            DataFrame with betting opportunities from get_best_opportunities
        results_df : pd.DataFrame
            DataFrame with match results, must have columns:
            - result: actual match result (1=home win, 0=draw, -1=away win)
            - flashscore_id or flashscore_id: match identifier
            - datetime: match date and time
        initial_bankroll : float, default=1000.0
            Initial bankroll for the backtest
        stake_method : str, default="fractional_kelly"
            Method to determine stake size:
            - "fractional_kelly": Use fractional Kelly criterion
            - "fixed_pct": Use fixed percentage of bankroll
            - "fixed_amount": Use fixed amount
        max_stake_pct : float, default=0.1
            Maximum stake as percentage of bankroll (0.1 = 10%)
        min_bankroll_pct : float, default=0.1
            Minimum bankroll percentage to continue betting (0.1 = 10%)

        Returns:
        --------
        pd.DataFrame
            DataFrame with backtest results
        """
        # Prepare results DataFrame
        if "flashscore_id" in results_df.columns:
            id_column = "flashscore_id"
        else:
            id_column = "flashscore_id"

        # Merge opportunities with results
        backtest = opportunities.merge(
            results_df.reset_index()[[id_column, "result", "datetime"]],
            left_on="flashscore_id",
            right_on=id_column,
        )

        # Set datetime as index and sort
        backtest.set_index("datetime", inplace=True)
        backtest.sort_index(inplace=True)

        # Check if bet was correct
        backtest["correct"] = backtest["result"] == backtest["outcome_in_model"]

        # Initialize backtest variables
        bankroll = initial_bankroll
        min_bankroll = initial_bankroll * min_bankroll_pct
        bets = []
        cum_bankroll = []
        profits = []
        roi = []
        drawdown = []
        max_bankroll = initial_bankroll

        # Run backtest
        for _, row in backtest.iterrows():
            # Skip if bankroll is below minimum
            if bankroll < min_bankroll:
                bets.append(0)
                cum_bankroll.append(bankroll)
                profits.append(0)
                roi.append(0)
                drawdown.append((max_bankroll - bankroll) / max_bankroll)
                continue

            # Calculate bet size based on stake method
            if stake_method == "fractional_kelly":
                bet_size = row["fractional_kelly"] * bankroll
            elif stake_method == "fixed_pct":
                bet_size = bankroll * max_stake_pct
            else:  # fixed_amount
                bet_size = max_stake_pct * initial_bankroll

            # Cap bet size at maximum percentage of bankroll
            bet_size = min(bet_size, bankroll * max_stake_pct)

            # Record bet
            bets.append(bet_size)
            cum_bankroll.append(bankroll)

            # Update bankroll based on bet result
            if row["correct"]:
                profit = bet_size * (row["bookmaker_odds"] - 1)
                bankroll += profit
            else:
                profit = -bet_size
                bankroll += profit

            # Update metrics
            profits.append(profit)
            roi.append(profit / bet_size if bet_size > 0 else 0)
            max_bankroll = max(max_bankroll, bankroll)
            drawdown.append(
                (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
            )

        # Add metrics to backtest DataFrame
        backtest["bet_size"] = bets
        backtest["cum_bankroll"] = cum_bankroll
        backtest["profit"] = profits
        backtest["roi"] = roi
        backtest["drawdown"] = drawdown
        backtest["cum_profit"] = backtest["profit"].cumsum()
        backtest["cum_roi"] = backtest["profit"].cumsum() / initial_bankroll

        return backtest

    def plot_backtest_results(
        self,
        backtest: pd.DataFrame,
        title: str = "Backtest Results",
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """
        Plot backtest results.

        Parameters:
        -----------
        backtest : pd.DataFrame
            DataFrame with backtest results from run_backtest
        title : str, default="Backtest Results"
            Title for the plots
        figsize : Tuple[int, int], default=(15, 10)
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        # Plot cumulative profit
        backtest["cum_profit"].plot(ax=axes[0, 0], title="Cumulative Profit")
        axes[0, 0].axhline(y=0, color="r", linestyle="-", alpha=0.3)
        axes[0, 0].set_ylabel("Profit")

        # Plot bankroll over time
        backtest["cum_bankroll"].plot(ax=axes[0, 1], title="Bankroll Over Time")
        axes[0, 1].set_ylabel("Bankroll")

        # Plot drawdown
        backtest["drawdown"].plot(ax=axes[1, 0], title="Drawdown")
        axes[1, 0].set_ylabel("Drawdown %")

        # Plot EV vs actual profit scatter
        axes[1, 1].scatter(backtest["ev"], backtest["profit"], alpha=0.5)
        axes[1, 1].set_title("Expected Value vs. Actual Profit")
        axes[1, 1].set_xlabel("Expected Value")
        axes[1, 1].set_ylabel("Actual Profit")
        axes[1, 1].axhline(y=0, color="r", linestyle="-", alpha=0.3)
        axes[1, 1].axvline(x=0, color="r", linestyle="-", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def calculate_performance_metrics(self, backtest: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Parameters:
        -----------
        backtest : pd.DataFrame
            DataFrame with backtest results from run_backtest

        Returns:
        --------
        Dict[str, float]
            Dictionary with performance metrics
        """
        if backtest.empty:
            return {
                "total_bets": 0,
                "win_rate": 0,
                "total_profit": 0,
                "roi": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "avg_ev": 0,
                "avg_odds": 0,
            }

        # Calculate metrics
        total_bets = len(backtest)
        win_rate = backtest["correct"].mean() if total_bets > 0 else 0
        total_profit = backtest["profit"].sum()
        roi = (
            total_profit / backtest["bet_size"].sum()
            if backtest["bet_size"].sum() > 0
            else 0
        )
        max_drawdown = backtest["drawdown"].max()

        # Calculate Sharpe ratio (annualized)
        daily_returns = backtest["profit"].groupby(backtest.index.date).sum()
        sharpe_ratio = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if len(daily_returns) > 1 and daily_returns.std() > 0
            else 0
        )

        # Calculate average EV and odds
        avg_ev = backtest["ev"].mean()
        avg_odds = backtest["bookmaker_odds"].mean()

        return {
            "total_bets": total_bets,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "roi": roi,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_ev": avg_ev,
            "avg_odds": avg_odds,
        }

    def prepare_model_probabilities(
        self, model_probabilities: pd.DataFrame, model_name: str = None
    ) -> pd.DataFrame:
        """
        Prepare model probabilities for evaluation.

        This method handles different formats of model probabilities:
        1. Multi-index DataFrame from ensemble.get_model_probabilities()
        2. Already unstacked DataFrame with home_prob, draw_prob, away_prob columns

        Parameters:
        -----------
        model_probabilities : pd.DataFrame
            DataFrame with model probabilities
        model_name : str, optional
            Name of the model to use (required if model_probabilities is from ensemble)

        Returns:
        --------
        pd.DataFrame
            DataFrame with model probabilities in the format expected by evaluate_opportunities
        """
        # Check if model_probabilities is already in the right format
        expected_columns = {"home_prob", "draw_prob", "away_prob"}
        if set(model_probabilities.columns).issuperset(expected_columns):
            return model_probabilities

        # Check if model_probabilities is from ensemble.get_model_probabilities()
        if (
            isinstance(model_probabilities.index, pd.MultiIndex)
            and model_name is not None
        ):
            # Extract the model's probabilities and unstack
            if model_name in model_probabilities.columns:
                return model_probabilities[model_name].unstack()
            else:
                raise ValueError(
                    f"Model '{model_name}' not found in model_probabilities"
                )

        # If we get here, the format is not recognized
        raise ValueError(
            "model_probabilities must either have columns ['home_prob', 'draw_prob', 'away_prob'] "
            "or be a multi-index DataFrame from ensemble.get_model_probabilities() with model_name specified"
        )


# %%Example usage
if __name__ == "__main__":
    # Load data
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        league="Kvindeligaen Women",
        # seasons=["2024/2025"],
    )
    bookmaker_odds = loader.load_odds(df.index)

    # Split data into train and test sets
    train_df = df.query("datetime < '2024-08-01'")
    test_df = df.query("datetime >= '2024-08-01'")

    # Prepare training data
    weights = dixon_coles_weights(train_df.datetime, xi=0.0000018)
    Z_train = (
        train_df[["home_goals", "away_goals"]]
        if "home_goals" in train_df.columns
        else None
    )
    Z_test = (
        test_df[["home_goals", "away_goals"]]
        if "home_goals" in test_df.columns
        else None
    )

    X_train = train_df[["home_team", "away_team"]]
    y_train = train_df["goal_difference"]
    X_test = test_df[["home_team", "away_team"]]
    y_test = test_df["goal_difference"]

    # Create and fit ensemble model
    model_names = ["bradley_terry", "gssd", "prp", "toor", "zsd"]
    ensemble = Ensemble(model_names)
    ensemble.fit(X_train, y_train, Z_train, ratings_weights=weights)

    # Get model predictions
    model_spreads = ensemble.get_model_spreads(X_test)
    model_probabilities = ensemble.get_model_probabilities(X_test)

    # Create evaluator with quarter Kelly (more conservative)
    evaluator = OddsEvaluator(fraction=0.25)

    # Compare different models
    results = {}
    for model_name in model_names + ["ens"]:
        # Prepare model probabilities
        prepared_probs = evaluator.prepare_model_probabilities(
            model_probabilities, model_name
        )

        # Evaluate opportunities
        opportunities = evaluator.evaluate_opportunities(bookmaker_odds, prepared_probs)

        # Get best opportunities with positive EV
        best_opportunities = evaluator.get_best_opportunities(
            opportunities, min_ev=0.08, selection_method="max_ev"
        )

        # Run backtest
        backtest = evaluator.run_backtest(
            best_opportunities,
            test_df,
            initial_bankroll=1000.0,
            stake_method="fractional_kelly",
            max_stake_pct=0.05,  # More conservative max stake
        )

        # Store results
        results[model_name] = {
            "backtest": backtest,
            "metrics": evaluator.calculate_performance_metrics(backtest),
        }

        # Plot results
        evaluator.plot_backtest_results(
            backtest, title=f"Backtest Results - {model_name}"
        )

    # Compare model performance
    performance_df = pd.DataFrame(
        {
            model: metrics
            for model, metrics in {
                model: data["metrics"] for model, data in results.items()
            }.items()
        }
    ).T

    print("Model Performance Comparison:")
    print(
        performance_df[
            ["total_bets", "win_rate", "total_profit", "roi", "sharpe_ratio"]
        ]
    )

    # Plot comparison of cumulative profits
    plt.figure(figsize=(12, 8))
    for model_name, data in results.items():
        if not data["backtest"].empty:
            data["backtest"]["cum_profit"].plot(label=model_name)

    plt.title("Cumulative Profit Comparison")
    plt.xlabel("Date")
    plt.ylabel("Profit")
    plt.legend()
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.show()


# %%
