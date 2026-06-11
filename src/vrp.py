"""
vrp.py — Variance Risk Premium computation utilities
Replicates Carr & Wu (2009) methodology.
"""

import numpy as np
import pandas as pd


def realized_variance(returns: pd.Series, freq: str = "5min", annualize: bool = False) -> float:
    """
    Compute realized variance from high-frequency returns.

    Parameters
    ----------
    returns : pd.Series
        Intraday log returns at specified frequency, DatetimeIndex.
    freq : str
        Sampling frequency label (for documentation only; caller pre-samples).
    annualize : bool
        If True, multiply by 252 (trading days).

    Returns
    -------
    float
        Realized variance over the sample window.

    Notes
    -----
    Excludes overnight returns. No microstructure noise correction applied here;
    use realized_variance_subsampled() for the ZMA (2005) estimator.
    """
    rv = (returns ** 2).sum()
    if annualize:
        n_days = returns.index.normalize().nunique()
        rv = rv * (252 / n_days) if n_days > 0 else np.nan
    return float(rv)


def realized_variance_subsampled(returns: pd.Series, K: int = 5) -> float:
    """
    Zhang, Mykland & Aït-Sahalia (2005) two-scale realized variance estimator.
    Reduces microstructure noise bias present in 1-min RV.

    Parameters
    ----------
    returns : pd.Series
        Full-grid (e.g. 1-min) log returns.
    K : int
        Subsampling grid size (5 = approximate 5-min grid from 1-min data).

    Returns
    -------
    float
        Noise-corrected realized variance.
    """
    n = len(returns)
    # Slow scale (full grid)
    rv_slow = (returns ** 2).sum()
    # Fast scale (every K-th observation)
    rv_fast_grids = [
        (returns.iloc[k::K] ** 2).sum() for k in range(K)
    ]
    rv_fast = np.mean(rv_fast_grids)
    # ZMA estimator
    rv_zma = rv_slow / K - (1 - 1 / K) * rv_fast
    return float(rv_zma)


def model_free_implied_variance(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    spot: float,
    rate: float,
    tau: float,
) -> float:
    """
    Model-free implied variance via Britten-Jones & Neuberger (2000) / Bakshi-Madan.

    Approximates:
        IV² ≈ 2 ∫ [C(K)/K² dK  (K > F)]  +  2 ∫ [P(K)/K² dK  (K < F)]

    Parameters
    ----------
    strikes : array of shape (N,)
        Sorted strike prices.
    call_prices : array of shape (N,)
        OTM call prices (use NaN for ITM strikes).
    put_prices : array of shape (N,)
        OTM put prices (use NaN for ITM strikes).
    spot : float
        Current underlying price.
    rate : float
        Continuously compounded risk-free rate.
    tau : float
        Time to expiration in years.

    Returns
    -------
    float
        Model-free implied variance (annualized).
    """
    F = spot * np.exp(rate * tau)
    dk = np.gradient(strikes)

    iv2 = 0.0
    for i, K in enumerate(strikes):
        if K > F and not np.isnan(call_prices[i]):
            iv2 += 2 * call_prices[i] / (K ** 2) * dk[i]
        elif K < F and not np.isnan(put_prices[i]):
            iv2 += 2 * put_prices[i] / (K ** 2) * dk[i]

    iv2 = iv2 * np.exp(rate * tau) / tau
    return float(iv2)


def interpolate_constant_maturity(
    iv2_near: float,
    iv2_far: float,
    t_near: float,
    t_far: float,
    target_days: int = 30,
) -> float:
    """
    Linear interpolation to 30-day constant maturity IV².

    Parameters
    ----------
    iv2_near, iv2_far : float
        Model-free implied variances for near/far expiry.
    t_near, t_far : float
        Days to expiry for near and far contracts.
    target_days : int
        Target maturity in calendar days.

    Returns
    -------
    float
        Interpolated constant-maturity IV².
    """
    if t_near == t_far:
        return iv2_near
    w = (target_days - t_near) / (t_far - t_near)
    w = np.clip(w, 0, 1)
    return float((1 - w) * iv2_near + w * iv2_far)


def compute_vrp(iv2: float, rv2: float) -> float:
    """
    VRP = IV² − RV²  (Carr & Wu 2009, eq. 1).

    Both inputs must be in the same units (annualized variance or not).
    Positive VRP means the market prices variance insurance above actuarial cost.
    """
    return float(iv2 - rv2)


def build_vrp_panel(
    iv2_df: pd.DataFrame,
    rv2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align IV² and RV² DataFrames and compute VRP panel.

    Parameters
    ----------
    iv2_df : DataFrame, index=dates, columns=ticker
    rv2_df : DataFrame, index=dates, columns=ticker

    Returns
    -------
    DataFrame of VRP values, same shape as the intersection.
    """
    iv2, rv2 = iv2_df.align(rv2_df, join="inner")
    vrp = iv2 - rv2
    return vrp
