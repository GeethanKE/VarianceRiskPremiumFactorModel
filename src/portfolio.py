"""
portfolio.py — Quintile portfolio construction and evaluation.
Long/short VRP factor portfolio following Carr & Wu (2009).
"""

import numpy as np
import pandas as pd
from typing import Literal


def quintile_sort(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    n_portfolios: int = 5,
    weighting: Literal["value_weighted", "equal_weighted"] = "value_weighted",
    market_cap: pd.DataFrame = None,
    lag: int = 1,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Sort stocks into portfolios by signal, compute forward returns.

    Parameters
    ----------
    signal : DataFrame (T x N)
        Ranking signal (VRP). Higher = higher expected return per hypothesis.
    returns : DataFrame (T x N)
        Stock returns.
    n_portfolios : int
        Number of sort portfolios (5 = quintiles).
    weighting : str
        'value_weighted' requires market_cap; 'equal_weighted' does not.
    market_cap : DataFrame (T x N) or None
        Market capitalizations for VW weighting.
    lag : int
        Signal-to-return lag in periods.
    transaction_cost_bps : float
        Round-trip cost per trade in basis points (applied on rebalance).

    Returns
    -------
    port_returns : DataFrame (T x n_portfolios + 2)
        Columns: Q1 ... Q{n}, LS (long-short), LS_net (after tc).
    """
    if weighting == "value_weighted" and market_cap is None:
        raise ValueError("market_cap required for value-weighted portfolios.")

    dates = signal.index[:-lag]
    port_labels = [f"Q{i+1}" for i in range(n_portfolios)]
    records = []
    prev_weights = {label: pd.Series(dtype=float) for label in port_labels}

    for t_idx, date in enumerate(dates):
        fwd_date = returns.index[t_idx + lag]
        sig_t = signal.loc[date].dropna()
        ret_t = returns.loc[fwd_date].dropna()
        common = sig_t.index.intersection(ret_t.index)

        if len(common) < n_portfolios * 5:
            records.append({**{l: np.nan for l in port_labels}, "LS": np.nan, "LS_net": np.nan})
            continue

        sig_t = sig_t.loc[common]
        ret_t = ret_t.loc[common]

        quantiles = pd.qcut(sig_t, n_portfolios, labels=port_labels)
        row = {}
        tc_cost = 0.0

        for label in port_labels:
            members = quantiles[quantiles == label].index
            if weighting == "value_weighted":
                mc_t = market_cap.loc[date, members].dropna()
                members = members.intersection(mc_t.index)
                w = mc_t.loc[members] / mc_t.loc[members].sum()
            else:
                w = pd.Series(1 / len(members), index=members)

            r_port = (ret_t.loc[members] * w).sum()

            # Transaction cost on turnover
            if transaction_cost_bps > 0:
                prev_w = prev_weights[label].reindex(members).fillna(0)
                turnover = (w - prev_w).abs().sum() / 2
                tc_cost += turnover * (transaction_cost_bps / 10_000)

            prev_weights[label] = w
            row[label] = float(r_port)

        ls = row.get(f"Q{n_portfolios}", np.nan) - row.get("Q1", np.nan)
        row["LS"] = ls
        row["LS_net"] = ls - tc_cost
        records.append(row)

    port_returns = pd.DataFrame(records, index=dates)
    return port_returns


def summary_statistics(
    port_returns: pd.DataFrame,
    rf: pd.Series = None,
    freq: int = 12,
) -> pd.DataFrame:
    """
    Annualized return, volatility, Sharpe ratio, max drawdown per portfolio.

    Parameters
    ----------
    port_returns : DataFrame
        Monthly portfolio returns from quintile_sort().
    rf : Series or None
        Risk-free rate matched to port_returns dates.
    freq : int
        Periods per year (12 = monthly).

    Returns
    -------
    DataFrame with one row per portfolio column.
    """
    if rf is not None:
        excess = port_returns.subtract(rf, axis=0)
    else:
        excess = port_returns

    stats = pd.DataFrame(index=port_returns.columns)
    stats["ann_return"] = port_returns.mean() * freq
    stats["ann_vol"] = port_returns.std() * np.sqrt(freq)
    stats["sharpe"] = excess.mean() / port_returns.std() * np.sqrt(freq)
    stats["max_drawdown"] = port_returns.apply(_max_drawdown)
    stats["skewness"] = port_returns.skew()
    stats["kurtosis"] = port_returns.kurtosis()
    return stats.round(4)


def capm_alpha(
    port_returns: pd.Series,
    market_returns: pd.Series,
    rf: pd.Series = None,
    annualize: bool = True,
    freq: int = 12,
) -> dict:
    """
    OLS CAPM alpha and beta for a portfolio return series.

    Returns dict with keys: alpha, beta, t_alpha, r_squared.
    """
    common = port_returns.index.intersection(market_returns.index)
    r = port_returns.loc[common]
    m = market_returns.loc[common]

    if rf is not None:
        rf_c = rf.reindex(common).fillna(0)
        r = r - rf_c
        m = m - rf_c

    X = np.column_stack([np.ones(len(m)), m.values])
    Y = r.values
    coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    alpha, beta = coefs

    resid = Y - X @ coefs
    sigma2 = resid.var(ddof=2)
    se_alpha = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
    t_alpha = alpha / se_alpha
    ss_res = (resid ** 2).sum()
    ss_tot = ((Y - Y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    if annualize:
        alpha *= freq

    return {"alpha": alpha, "beta": beta, "t_alpha": t_alpha, "r_squared": r2}


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())
