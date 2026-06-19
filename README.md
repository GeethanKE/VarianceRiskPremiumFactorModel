# VRP Factor Model

Replication of **Carr & Wu (2009)** with **live market data**.

## How to run

1. Clone this repo
2. Open Terminal in the repo folder
3. Run:

```bash
pip3 install jupyter
jupyter notebook
```

4. Click **VRP_Factor_Model.ipynb**
5. Click **Kernel → Restart & Run All**

That's it. The notebook installs its own dependencies (Cell 1) and pulls
live S&P 500 + VIX data automatically (Cell 2). No separate data files needed.

## What it does

| Cell | What happens |
|------|-------------|
| 1 | Installs `yfinance`, `pandas`, `numpy`, `matplotlib`, `statsmodels` |
| 2 | Downloads 10 years of daily prices for 50 S&P 500 stocks + VIX |
| 3 | Computes IV², RV², and VRP for each stock each month |
| 4 | Runs Fama-MacBeth cross-sectional regressions by sub-period |
| 5 | Plots rolling 24-month λ₁ to visualise post-publication decay |
| 6 | Constructs VRP quintile portfolios, reports Sharpe / drawdown |
| 7 | Microstructure tests: RV frequency, crisis dependency, current snapshot |
| 8 | Saves summary dashboard as `vrp_dashboard.png` |

## Formula

```
VRP_{i,t} = IV²_{i,t} − RV²_{i,t}

r_{i,t+1} = λ₀ + λ₁·VRP_{i,t} + λ₂·β_{i,t} + λ₃·log(ME)_{i,t} + ε
```

## Data sources

- **Prices & returns**: Yahoo Finance via `yfinance` (free, real-time)
- **Implied variance proxy**: VIX² scaled by stock beta
- **Realised variance**: rolling 21-day sum of squared daily log returns

## License
MIT
