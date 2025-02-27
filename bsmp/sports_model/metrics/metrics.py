# %%
"""This file contains the implementation of the Metrics class for evaluating sports models."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from bsmp.data_models.data_loader import MatchDataLoader
from bsmp.sports_model.ensemble import NNLS, SimpleEnsemble
from bsmp.sports_model.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry, Poisson
from bsmp.sports_model.utils import dixon_coles_weights


class Metrics:
    """Class for calculating and storing metrics for sports models."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        probabilities: pd.DataFrame,
        true_values: pd.Series,
    ):
        """Initialize the Metrics class.

        Args:
            predictions: pd.DataFrame, predictions from the model
            probabilities: pd.DataFrame, probabilities from the model
            true_values: pd.Series, true values
        """
        self.predictions = predictions
        self.probabilities = probabilities
        self.true_values = true_values

    def regression_metrics(self, model_predictions):
        """Calculate regression metrics for a specific model's predictions."""
        mae = mean_absolute_error(self.true_values, model_predictions)
        mse = mean_squared_error(self.true_values, model_predictions)
        rmse = mse**0.5
        r2 = r2_score(self.true_values, model_predictions)

        return {
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Median Absolute Error": median_absolute_error(
                self.true_values, model_predictions
            ),
            "R-squared": r2,
        }

    def classification_metrics(self, probabilities, outcome):
        """Calculate classification metrics for a specific model's predictions."""
        probabilities_outcome = probabilities.map(lambda x: 1 if x > 0.5 else 0)
        if outcome == "home_win":
            true_classifications = pd.Series(0, index=self.true_values.index)
            true_classifications[self.true_values > 0] = 1  # Home win

        elif outcome == "draw":
            true_classifications = pd.Series(0, index=self.true_values.index)
            true_classifications[self.true_values == 0] = 1  # Draw

        elif outcome == "away_win":
            true_classifications = pd.Series(0, index=self.true_values.index)
            true_classifications[self.true_values < 0] = 1  # Away win

        else:
            raise ValueError("Invalid outcome")

        return {
            "Accuracy": accuracy_score(true_classifications, probabilities_outcome),
            "Log Loss": log_loss(true_classifications, probabilities),
            "Brier Score": brier_score_loss(true_classifications, probabilities),
        }

    def regression_metrics_to_df(self):
        """Create a DataFrame of metrics with model names as columns."""
        metrics_dict = {}

        for model_name in self.predictions.columns:
            model_predictions = self.predictions[model_name]
            regression = self.regression_metrics(model_predictions)

            # Combine regression
            all_metrics = {**regression}
            metrics_dict[model_name] = all_metrics

        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_dict)
        return metrics_df

    def classification_metrics_to_df(self, outcome):
        """Create a DataFrame of metrics with model names as columns."""
        metrics_dict = {}

        for model_name in self.probabilities.columns:
            model_predictions = self.probabilities[model_name]
            classification = self.classification_metrics(
                model_predictions, outcome=outcome
            )

            # Combine classification
            all_metrics = {**classification}
            metrics_dict[model_name] = all_metrics

        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_dict)
        return metrics_df


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
    weights = dixon_coles_weights(train_df.datetime, xi=0.018)

    home_team = "Kolding"
    away_team = "Sonderjyske"
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
    probabilities = pd.DataFrame(0.0, columns=model_list, index=test_df.index)
    outcome = "draw"

    for model_name, model in zip(model_list, models):
        model.fit(
            y=train_df["goal_difference"],
            X=train_df[["home_team", "away_team"]],
            Z=train_df[["home_goals", "away_goals"]],
            weights=weights,
        )
        preds = model.predict(X=test_df[["home_team", "away_team"]])
        predictions.loc[test_df.index, model_name] = preds
        probas = model.predict_proba(
            X=test_df[["home_team", "away_team"]], outcome=outcome
        )
        probabilities.loc[test_df.index, model_name] = probas

    metrics = Metrics(predictions, probabilities, test_df["goal_difference"])
    metrics_df = metrics.regression_metrics_to_df()
    classification_df = metrics.classification_metrics_to_df(outcome=outcome)

    metrics_df
    classification_df


# %%
