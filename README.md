# ⚡ VRP Factor Model — Real-Time Tick Stream (Alpaca)

Live tick-by-tick variance risk premium model using Alpaca's free WebSocket API.

## Setup (one time)

1. Sign up free at [alpaca.markets](https://alpaca.markets) → Paper Trading → Generate API Keys
2. Open `.env` in this folder and paste your keys:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

3. In Terminal:

```bash
pip3 install jupyterlab
jupyter lab
```

4. Open **VRP_Live_Alpaca.ipynb** → **Run → Run All Cells**

## What it does

| Cell | What happens |
|------|-------------|
| 1 | Installs `alpaca-py`, `pandas`, `numpy`, `matplotlib` |
| 2 | Loads keys from `.env`, connects to Alpaca |
| 3 | Seeds 90 days of historical bars, builds VRP functions |
| 4 | Opens WebSocket tick stream + live refreshing dashboard |

## Formula

```
VRP = IV² − RV²

IV²  = Parkinson (1980) high-low estimator — no options data needed
         IV² = mean[(ln(H/L))² / 4ln2] × 252

RV²  = rolling 21-bar realised variance from tick prices
         RV² = Σ(log return)² × 252/21
```

## Signals

| VRP | Signal |
|-----|--------|
| > +0.01 | 🟢 LONG |
| < −0.01 | 🔴 SHORT |
| between | ⚪ FLAT |

## Security

`.env` is in `.gitignore` — your keys will never be pushed to GitHub.
Never commit your keys or paste them in public.

## License
MIT
