"""
fama_macbeth.py — Cross-sectional Fama-MacBeth regression engine.
Replicates the estimation procedure in Carr & Wu (2009).
"""

import numpy as np
import pandas as pd
from typing import Optional


def newey_west(x: np.ndarray, lags: int = 6) -> np.ndarray:
    """
    Newey-West HAC covariance matrix for a time series of estimates.

    Parameters
    ----------
    x : array of shape (T, K)
        Time series of K coefficient estimates.
    lags : int
        Number of lags for bandwidth.

    Returns
    -------
    cov : array of shape (K, K)
        HAC covariance matrix.
    """
    T, K = x.shape
    xd = x - x.mean(axis=0)
    cov = xd.T @ xd / T
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        gamma = xd[lag:].T @ xd[:-lag] / T
        cov += w * (gamma + gamma.T)
    return cov


def fama_macbeth(
    returns: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    lag: int = 1,
    nw_lags: int = 6,
) -> pd.DataFrame:
    """
    Run Fama-MacBeth (1973) cross-sectional regressions.

    At each date t, regresses next-period returns on date-t factor exposures.
    Averages slope coefficients across time and computes Newey-West t-statistics.

    Parameters
    ----------
    returns : DataFrame, shape (T, N)
        Panel of stock returns. Index = dates, columns = tickers.
    factors : dict of str -> DataFrame
        Each value is a DataFrame of the same shape as `returns`.
        Keys become column names in output (e.g. "VRP", "beta", "log_me").
    lag : int
        Return horizon in periods (default 1 = next-month return).
    nw_lags : int
        Newey-West lag truncation.

    Returns
    -------
    results : DataFrame
        Columns: [lambda, t_stat, p_value, n_avg] for intercept + each factor.
    """
    dates = returns.index[:-lag]
    factor_names = list(factors.keys())
    K = len(factor_names) + 1  # +1 for intercept

    lambdas = np.full((len(dates), K), np.nan)

    for t_idx, date in enumerate(dates):
        fwd_date = returns.index[t_idx + lag]
        y = returns.loc[fwd_date].dropna()

        rows = []
        for ticker in y.index:
            row = [1.0]  # intercept
            valid = True
            for fname in factor_names:
                df = factors[fname]
                if date in df.index and ticker in df.columns:
                    val = df.loc[date, ticker]
                    if np.isfinite(val):
                        row.append(val)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid:
                rows.append((ticker, row))

        if len(rows) < K + 5:
            continue

        tickers_t = [r[0] for r in rows]
        X = np.array([r[1] for r in rows])
        Y = y.loc[tickers_t].values

        try:
            coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
            lambdas[t_idx] = coefs
        except np.linalg.LinAlgError:
            continue

    valid_mask = np.isfinite(lambdas[:, 0])
    lambdas_clean = lambdas[valid_mask]
    T = lambdas_clean.shape[0]

    if T < 10:
        raise ValueError(f"Too few valid cross-sections: {T}")

    lam_mean = lambdas_clean.mean(axis=0)
    nw_cov = newey_west(lambdas_clean, lags=nw_lags)
    se = np.sqrt(np.diag(nw_cov) / T)
    t_stats = lam_mean / se
    p_values = 2 * (1 - _normal_cdf(np.abs(t_stats)))

    col_names = ["intercept"] + factor_names
    results = pd.DataFrame(
        {
            "lambda": lam_mean,
            "se": se,
            "t_stat": t_stats,
            "p_value": p_values,
            "n_periods": T,
        },
        index=col_names,
    )
    return results


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via error function."""
    return 0.5 * (1 + np.sign(x) * (1 - np.exp(-x**2 * (4/np.pi + 0.147 * x**2) /
                                                  (1 + 0.147 * x**2))))


def rolling_fama_macbeth(
    returns: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    window: Optional[int] = None,
    nw_lags: int = 6,
) -> pd.DataFrame:
    """
    Run Fama-MacBeth over expanding or rolling sub-windows.
    Useful for documenting temporal decay of the VRP premium.

    Parameters
    ----------
    window : int or None
        Rolling window in months. None = expanding (cumulative from start).

    Returns
    -------
    DataFrame with one row per end-date in dates, columns = factor names + stats.
    """
    dates = returns.index
    records = []

    for end_idx in range(24, len(dates)):
        start_idx = max(0, end_idx - window) if window else 0
        sub_returns = returns.iloc[start_idx:end_idx]
        sub_factors = {k: v.iloc[start_idx:end_idx] for k, v in factors.items()}

        try:
            res = fama_macbeth(sub_returns, sub_factors, nw_lags=nw_lags)
            row = {"end_date": dates[end_idx]}
            for col in res.index:
                row[f"{col}_lambda"] = res.loc[col, "lambda"]
                row[f"{col}_tstat"] = res.loc[col, "t_stat"]
            records.append(row)
        except (ValueError, np.linalg.LinAlgError):
            continue

    return pd.DataFrame(records).set_index("end_date")
