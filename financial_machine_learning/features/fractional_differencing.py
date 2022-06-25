"""
Implements fractional differencing.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_weights(order: float, lags: int) -> List[float]:
    """Calculate the weights from the series expansion of the differencing operator
    for orders d up to lags coefficients.

    Args:
        order (float): The exponent of (1-B)^d
        lags (int): The number of coefficients of the infinite taylor series

    Returns:
        List[float]: A list of weights with the i-th position being the weight w_i.
    """
    weights = [1]
    for k in range(1,lags):
        weights.append(-weights[-1]*((order-k+1))/k)
    weights = np.array(weights).reshape(-1,1)
    return weights

def plot_weights(d_range: Tuple[float, float], lags: int, number_plots: int) -> None:
    """Plots the weights by its coefficient position for a range of values for d.

    Args:
        d_range (tuple): The lower and upper value of d's to plot.
        lags (int): The cut-off of the infinite series
        number_plots (int): The number of different d's to plot.
    """
    weights = pd.DataFrame(np.zeros((lags, number_plots)))
    interval = np.linspace(d_range[0], d_range[1], number_plots)
    for i, diff_order in enumerate(interval):
        weights[i] = get_weights(diff_order, lags)
    weights.columns = [round(x, 2) for x in interval]
    weights.plot(figsize=(15, 6))
    plt.legend(title='Order of differencing')
    plt.title('Lag coefficients for various orders of differencing')
    plt.xlabel('lag coefficients')
    plt.show()

def differencing(series: pd.Series, order: float, lag_cutoff: int) -> List[float]:
    """Returns the fractional difference of a series for a given order and cuttoff
    of the infinite series

    Args:
        series (pd.Series): Input
        order (float): The order of difference (1-B)^d
        lag_cutoff (int): The cuttoff for the infinite series expansion

    Returns:
        List[float]: The fractional difference of the series.
    """
    weights = get_weights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:]

def cutoff_find(order: float, tau: float=1e-4, start_lags: int=1) -> int:
    """Instead of cutting off at the i-th position in the infinite
    series expansion of (1-B)^d, we use can calculate the number of
    lags needed to get a weight w_i < tau.

    Args:
        order (float): The order of difference (1-B)^d
        tau (float): Threshold value to drop non-significant weights. Defaults to 1e-4.
        start_lags (int): A minimal number of lags we take. Defaults to 1.

    Returns:
        int: Returns the number of lags needed to reach the cutoff.
    """
    val = np.inf
    lags = start_lags
    while abs(val) > tau:
        weights = get_weights(order, lags)
        val = weights[len(weights) - 1]
        lags += 1
    return lags

def differencing_tau(series: pd.Series, order: float, tau: float=1e-4) -> List[float]:
    """Returns the fractional difference of a series for a given order
    of the infinite series stopping once the weights are smaller than tau.

    Args:
        series (pd.Series): Input
        order (float): The order of difference (1-B)^d
        tau (float): Threshold value to drop non-significant weights. Defaults to 1e-4.

    Returns:
        List[float]: The fractional difference of the series.
    """
    lag_cutoff = cutoff_find(order, tau, 1) #finding lag cutoff with tau
    weights = get_weights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:]
