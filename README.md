# VRP Cross-Sectional Factor Model

Replication and extension of **Carr & Wu (2009)** — *"Variance Risk Premiums"* — using post-2015 data. Documents post-publication alpha decay and investigates microstructure explanations.

## Overview

The variance risk premium (VRP) is defined as the difference between implied and realized variance:

```
VRP_{i,t} = IV²_{i,t} − RV²_{i,t}
```

where `IV²` is the model-free implied variance from 30-day ATM options (OptionMetrics) and `RV²` is realized variance computed from 5-minute intraday returns (TAQ).

The core question: **does the cross-sectional VRP signal still predict returns after publication?**

---

## Results summary

| Period | λ̂₁ (VRP slope) | t-stat | Signal |
|---|---|---|---|
| 1996–2009 (in-sample) | +0.0184 | 4.72 | ✅ Strong |
| 2010–2015 (OOS) | +0.0121 | 2.84 | ✅ Moderate |
| 2016–2022 (post-pub) | +0.0061 | 1.41 | ⚠️ Weak |
| 2023–2025 (extended) | −0.0018 | −0.39 | ❌ Noise |

**Key finding:** ~67% decay in the VRP premium post-publication. Net-of-cost alpha turns negative after 5bps round-trip transaction costs.

---

## Repo structure

```
vrp-factor-model/
├── README.md
├── LICENSE
├── requirements.txt
├── dashboard/
│   └── index.html              # Interactive research dashboard
├── notebooks/
│   ├── 01_data_pipeline.ipynb  # IV and RV construction
│   ├── 02_fama_macbeth.ipynb   # Cross-sectional regressions
│   ├── 03_portfolio_sorts.ipynb# Quintile portfolio construction
│   └── 04_decay_analysis.ipynb # Post-publication decay & microstructure
├── src/
│   ├── vrp.py                  # VRP computation utilities
│   ├── fama_macbeth.py         # Fama-MacBeth regression engine
│   └── portfolio.py            # Portfolio sort and evaluation
└── data/
    └── .gitkeep                # Raw data not included (see Data section)
```

---

## Methodology

### 1. VRP construction

- **Implied variance (IV²):** Model-free implied variance à la Britten-Jones & Neuberger (2000), integrated over the full strike surface from OptionMetrics. 30-day constant maturity via interpolation.
- **Realized variance (RV²):** Sum of squared 5-minute log returns over the prior 30 calendar days from TAQ data. Overnight returns excluded. Microstructure noise correction via Zhang, Mykland & Aït-Sahalia (2005).

### 2. Universe

- S&P 500 constituents at each month-end (point-in-time, no survivorship bias)
- Liquidity filter: ≥500 open interest contracts in the front two expirations
- Earnings exclusion: drop observations within ±3 days of scheduled earnings

### 3. Fama-MacBeth regression

Monthly cross-sectional regression:

```
r_{i,t+1} = λ₀_t + λ₁_t · VRP_{i,t} + λ₂_t · β_{i,t} + λ₃_t · log(ME)_{i,t} + ε_{i,t+1}
```

Time-series average of monthly slope coefficients; Newey-West HAC standard errors (6 lags).

### 4. Portfolio sorts

- Quintile sorts on VRP each month-end
- Value-weighted returns within quintiles
- Long Q5 (high VRP), short Q1 (low VRP)
- Evaluated on raw return, CAPM alpha, FF3 alpha, FF5 alpha

### 5. Microstructure channels investigated

| Channel | Method |
|---|---|
| Bid-ask bounce in IV | Compare NBBO mid vs. transaction price IV |
| Crowding / vol-selling AUM | Correlate premium with CBOE Put/Call ratio, vol-fund flows |
| Jump asymmetry | Separate continuous vs. jump component of RV (BNS test) |
| Sampling frequency | Robustness: 1-min vs 5-min vs 15-min RV |
| Limits to arbitrage | Interact VRP with Amihud illiquidity, short interest |

---

## Data

Raw data is **not included** due to licensing restrictions.

| Dataset | Source | Notes |
|---|---|---|
| Option prices / IV surface | OptionMetrics Ivy DB | University/institutional license |
| Intraday tick data | NYSE TAQ | WRDS subscription |
| Daily returns, market cap | CRSP | WRDS subscription |
| Risk-free rate | Ken French Data Library | Free |
| Factor returns (FF3/FF5) | Ken French Data Library | Free |

To reproduce, place data in `data/` following the schema documented in `notebooks/01_data_pipeline.ipynb`.

---

## Replication notes

This study replicates **Carr & Wu (2009), *Journal of Financial Economics* 90(1)**. Key differences:

- Extended sample through 2025
- Point-in-time S&P 500 membership (no survivorship bias)
- HAC standard errors vs. OLS in original
- Additional controls: BM ratio, momentum (12-1), short-term reversal (1-0)

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{carr2009variance,
  title={Variance risk premiums},
  author={Carr, Peter and Wu, Liuren},
  journal={The Review of Financial Studies},
  volume={22},
  number={3},
  pages={1311--1341},
  year={2009}
}
```

---

## License

MIT — see `LICENSE`.
