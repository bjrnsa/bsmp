from datetime import date
from typing import List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def dixon_coles_weights(
    dates: Union[List[date], pd.Series], xi: float = 0.0018, base_date: date = None
) -> NDArray:
    """
    Calculates a decay curve based on the algorithm given by Dixon and Coles in their paper.

    Parameters
    ----------
    dates : Union[List[date], pd.Series]
        A list or pd.Series of dates to calculate weights for.
    xi : float, optional
        Controls the steepness of the decay curve. Defaults to 0.0018.
    base_date : date, optional
        The base date to start the decay from. If set to None, it uses the maximum date from the dates list. Defaults to None.

    Returns
    -------
    NDArray
        An array of weights corresponding to the input dates.
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights
