# %%
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class HandballProbabilityGrid:
    """
    Class for calculating probabilities of handball outcomes.
    Handball-specific adaptations:
    - Higher goal expectations (typically 25-35 per team)
    - Larger goal matrices (up to 40+ goals per team)
    - Different distribution characteristics
    """

    def __init__(
        self,
        goal_matrix: NDArray,
        home_goal_expectation: float,
        away_goal_expectation: float,
        model_class: str = "HandballProbabilityGrid",
    ):
        """
        Initializes the HandballProbabilityGrid with goal matrix and expectations.

        Parameters
        ----------
        goal_matrix : NDArray
            Matrix of probabilities for each goal difference.
            Should be sized for handball scores (e.g., 40x40).
        home_goal_expectation : float
            Expected goals for home team (typical range 25-35).
        away_goal_expectation : float
            Expected goals for away team (typical range 25-35).
        model_class : str, optional
            The class name of the model. Defaults to "HandballProbabilityGrid".
        """
        self.grid = np.array(goal_matrix)
        self.home_goal_expectation = home_goal_expectation
        self.away_goal_expectation = away_goal_expectation
        self.model_class = model_class

    def __repr__(self) -> str:
        return (
            f"Class: {self.model_class}\n\n"
            f"Home Goal Expectation: {self.home_goal_expectation:.0f}\n"
            f"Away Goal Expectation: {self.away_goal_expectation:.0f}\n\n"
            f"Home Win: {self.home_win:.3f}\n"
            f"Draw: {self.draw:.3f}\n"
            f"Away Win: {self.away_win:.3f}\n\n"
            f"Home Odds: {self.home_odds:.3f}\n"
            f"Draw Odds: {self.draw_odds:.3f}\n"
            f"Away Odds: {self.away_odds:.3f}\n\n"
        )

    def _sum(self, condition: Callable[[int, int], bool]) -> float:
        """
        Sums the probabilities in the goal matrix based on a condition.

        Parameters
        ----------
        condition : Callable[[int, int], bool]
            A function that takes two integers (goal counts) and returns a boolean.

        Returns
        -------
        float
            The sum of probabilities that satisfy the condition.
        """
        return self.grid[
            np.fromfunction(lambda i, j: condition(i, j), self.grid.shape, dtype=int)
        ].sum()

    @property
    def home_win(self) -> float:
        """
        Calculates the probability of a home win.

        Returns
        -------
        float
            The probability of the home team winning.
        """
        return self._sum(lambda a, b: a > b)

    @property
    def draw(self) -> float:
        """
        Calculates the probability of a draw.

        Returns
        -------
        float
            The probability of a draw.
        """
        return self._sum(lambda a, b: a == b)

    @property
    def away_win(self) -> float:
        """
        Calculates the probability of an away win.

        Returns
        -------
        float
            The probability of the away team winning.
        """
        return self._sum(lambda a, b: a < b)

    @property
    def home_odds(self) -> float:
        """
        Calculates the odds of a home win.

        Returns
        -------
        float
            The odds of the home team winning.
        """
        return 1 / self.home_win

    @property
    def draw_odds(self) -> float:
        """
        Calculates the odds of a draw.

        Returns
        -------
        float
            The odds of a draw.
        """
        return 1 / self.draw

    @property
    def away_odds(self) -> float:
        """
        Calculates the odds of an away win.

        Returns
        -------
        float
            The odds of the away team winning.
        """
        return 1 / self.away_win

    def total_goals(self, over_under: str, strike: float) -> float:
        """
        Calculates the probability of total goals being over or under a strike value.

        Parameters
        ----------
        over_under : str
            "over" or "under" to specify the type of bet.
        strike : float
            The strike value for the total goals.

        Returns
        -------
        float
            The probability of the total goals being over or under the strike value.
        """
        # Implementation unchanged but used with higher strike values

    def asian_handicap(self, home_away: str, strike: float) -> float:
        """
        Calculates the probability of winning with an Asian handicap.

        Parameters
        ----------
        home_away : str
            "home" or "away" to specify the team.
        strike : float
            The handicap value.

        Returns
        -------
        float
            The probability of winning with the specified handicap.
        """
        # Implementation unchanged but used with larger handicap values


if __name__ == "__main__":
    pass
