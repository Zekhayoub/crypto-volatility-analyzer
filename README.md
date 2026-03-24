# Crypto Volatility & Microstructure Analyzer

Quantitative volatility modeling and signal detection for digital asset markets.

A modular Python pipeline that ingests 8-hourly crypto data from Binance and
Hyperliquid, computes professional market-making analytics (multi-estimator
volatility, funding rate signals, order flow imbalance), detects volatility
regimes via GJR-GARCH and HMM, and simulates a funding rate arbitrage strategy
with realistic execution costs.

> **Status:** Active development — core pipeline functional, backtest and documentation in progress.

> **Built for:** Quantitative analyst roles at crypto market makers.

---

## Methodology

```
Binance + Hyperliquid → 8h Master DataFrame
    → Feature Engineering (vol, funding pctile, OFI, correlation, drawdown)
    → GJR-GARCH Volatility Modeling
    → HMM Regime Detection (expanding window, BIC selection)
    → Signal Analysis (funding mean-reversion, liquidation cascades)
    → Funding Rate Arbitrage Backtest (full PnL decomposition)
```

---

## Why This Project Matters

- **Market maker perspective:** every metric is chosen because a trading desk
  uses it daily — funding percentiles, OFI, volatility regime, correlation asymmetry.
- **Realistic backtest:** funding arb simulation with maker/taker fees, dynamic
  execution cost modeling, margin monitoring, hurdle rate, and full PnL decomposition.
- **Multi-venue:** compares Binance (CEX) and Hyperliquid (DEX) funding rates,
  directly addressing the expansion of market makers into Perpetual DEX markets.
- **Rust integration:** Garman-Klass volatility via PyO3 with zero-copy FFI,
  demonstrating the Python→Rust migration path for compute-intensive functions.

---



## Architecture

```
crypto-volatility-analyzer/
    config/config.yaml       # All parameters 
    src/
        config.py            # Config loader + periods() helper
        ingestion.py         # Binance + Hyperliquid multi-source ingestion
        features.py          # 14 feature functions (vol, funding, OFI, correlation)
        volatility.py        # GJR-GARCH + EWMA fallback + persistence check
        regime.py            # HMM (expanding window, BIC, cov fallback)
        signals.py           # Signal analysis + funding arb backtest
        pipeline.py          # End-to-end orchestrator 
    rust/                    # PyO3 Garman-Klass module
        Cargo.toml
        pyproject.toml
        src/
            lib.rs           # PyO3 entry point, zero-copy FFI
            volatility.rs    # Garman-Klass rolling computation
    notebooks/
        eda.ipynb            # EDA + signals + backtest visualization
    figures/                 # Publication-quality figures
    results/                 # JSON/CSV outputs (GARCH params, backtest, signals)
    models/                  # Serialized HMM model (joblib)
```

---

## Technical Highlights

### GJR-GARCH
- Asymmetric volatility model confirming the leverage effect (γ > 0) — negative shocks amplify volatility more than positive shocks due to liquidation cascades.
- Automatic EWMA fallback when persistence ≥ 0.99 (IGARCH regime).

### HMM Regime Detection
- 3-state Hidden Markov Model (risk-on, risk-off, crisis) fitted with expanding window and BIC model selection.
- Mitigation of look-ahead bias through strict temporal ordering.

### Funding Rate Arbitrage Backtest
- Full PnL decomposition: funding captured, delta exposure, execution costs, hurdle rate.
- Execution cost simulation incorporating orderbook depth.
- Margin monitoring throughout the simulation.


### Rust Integration

The Garman-Klass volatility computation is optionally accelerated via a Rust
module (PyO3/maturin). The pipeline includes an automatic fallback — if the
Rust module is not built, the equivalent Python implementation is used.
```
rust/
    Cargo.toml
    pyproject.toml
    src/
        lib.rs           # PyO3 entry point, zero-copy FFI
        volatility.rs    # Garman-Klass rolling computation
```

The architecture is deliberately modular: each Rust function is exposed as
a standalone Python-callable module via PyO3. This means additional
compute-intensive functions (e.g., rolling correlation, VPIN, orderbook
imbalance metrics) can be migrated to Rust incrementally without modifying
the existing Python codebase.

Build: `cd rust && maturin develop --release`

---

## Data

All data from public APIs (no keys required for Binance public endpoints).

| Source | Endpoint | Data | Frequency |
|--------|----------|------|-----------|
| Binance Spot | /api/v3/klines | BTC/ETH OHLCV + taker buy vol | 8h native |
| Binance Futures | /fapi/v1/fundingRate | Funding rates | 8h native |
| Binance Futures | /futures/data/openInterestHist | Open interest | Daily → 8h |
| Hyperliquid | /info | DEX funding rates (1h → scaled to 8h) | 1h native |

**Why 8h?** The funding rate — the most reactive signal in crypto — is natively
8-hourly. Aggregating to daily destroys information. This is the natural granularity
for funding arbitrage analysis.

---

## How to Run

```bash
git clone https://github.com/Zekhayoub/crypto-volatility-analyzer.git
cd crypto-volatility-analyzer
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

# Full pipeline
python -m src.pipeline

# Skip data download (use cached)
python -m src.pipeline --skip-ingestion

# Optional: build Rust module
cd rust && maturin develop --release && cd ..
```

---

## Limitations & Known Risks

### Data Limitations
- **No L1/L2 order book data.** OFI from klines is an approximation of true order flow toxicity.
- **Calendar time (8h).** Volume time sampling would better capture information arrival during high-activity periods but requires tick-level data.
- **Garman-Klass assumes continuous diffusion.** Liquidation wicks create jump discontinuities.

### Model Limitations
- **HMM is fitted in-sample** (Viterbi over full dataset). True production backtesting requires Walk-Forward Optimization with periodic refitting.
- **HMM transition matrix is time-homogeneous.** Transition probabilities should be conditioned by exogenous variables in production.
- **GJR-GARCH may be explosive** (IGARCH) during extreme stress. Persistence ≥ 0.99 triggers automatic EWMA fallback.

### Crypto-Specific Risks
- **Auto-Deleveraging (ADL):** exchange can forcibly close profitable positions.
- **Stablecoin peg assumption:** USDT/USDC peg stability is assumed.
- **Basis risk:** spot and perpetual prices can diverge beyond funding-implied levels during extreme stress.
- **Exchange counterparty risk:** FTX collapse (Nov 2022) demonstrated that exchange solvency is a non-trivial risk factor.

### Backtest Assumptions
- **Execution assumes maker orders.** Taker scenario shown for comparison.
- **Market impact assumed negligible** at simulated position sizes.

---

## Tech Stack

Python · Pandas · Polars · NumPy · Rust (PyO3/maturin) · requests · arch (GARCH) ·
hmmlearn · scikit-learn · scipy · statsmodels · matplotlib · seaborn · joblib

---

## References

1. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
2. Glosten, Jagannathan, Runkle (1993). "On the Relation between Expected Value and Volatility" (GJR)
3. Garman, M. & Klass, M. (1980). "On the Estimation of Security Price Volatilities"
4. Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return"
5. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"