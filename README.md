# Crypto Trading Performance & Predictive Modelling

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-yellow?style=for-the-badge&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A complete end-to-end study — from raw on-chain data to a deployable predictive model**

*By [Maneet Gupta](https://linkedin.com/in/maneet-gupta) · Ghaziabad, Uttar Pradesh, India*

</div>

---

##  Project at a Glance

| Metric | Value |
|---|---|
|  Raw fill executions | **211,224** |
|  Coins traded | **246** |
|  Accounts | **32** |
|  Date range | **May 2023 – May 2025** |
|  Final model | **XGBoost** |
|  Test ROC-AUC | **0.91** |
|  Win rate at threshold 0.65 | **61.1%** (vs 24.7% baseline) |

This project analyses real on-chain cryptocurrency trading data to answer two questions: **does market sentiment predict trade profitability**, and **can a machine-learning model reliably identify which trades are worth taking?**

> **Key discovery:** The first naive model showed AUC = 0.99. This was entirely fake — data leakage caused by three compounding errors. Fixing it required restructuring the entire dataset from scratch.


---

## 🔄 Pipeline Overview

```
Raw CSVs
   │
   ▼
1. Data Loading & Merging ──────► Timestamp parsing, Fear & Greed join
   │
   ▼
2. Exploratory Data Analysis ───► Distributions, correlation, sentiment performance
   │
   ▼
3. Baseline Model (Fill-Level) ─► AUC = 0.99 → LEAKAGE DETECTED
   │
   ▼
4. Leakage Diagnosis ───────────► Remove 6 leaky features + temporal split → AUC = 0.47
   │
   ▼
5. Trade-Level Reconstruction ──► VWAP aggregation + alpha feature engineering
   │
   ▼
6. Final XGBoost Model ─────────► AUC = 0.91, Win Rate = 61.1% at threshold 0.65
```

---

## 📊 Data Description

### Fill-Level Dataset (`historical_data.csv`)

When a trader places a large order, the exchange fills it across multiple price levels. Each sub-execution is a **fill**. The dataset contains 211,224 such fills.

| Column | Type | Description |
|---|---|---|
| `Account` | Text | On-chain wallet address (32 unique) |
| `Coin` | Text | Asset traded: BTC, ETH, SOL, etc. (246 unique) |
| `Execution Price` | Numeric | Exact fill price |
| `Size USD` | Numeric | Dollar notional of this fill |
| `Size Tokens` | Numeric | Tokens = Size USD / Execution Price |
| `Side` | BUY / SELL | BUY opens positions; SELL closes and realises PnL |
| `Timestamp IST` | DateTime | Execution time (UTC+5:30) |
| `Closed PnL` | Numeric | Realised P&L — always 0 for BUY fills |
| `Order ID` | Integer | Groups fills belonging to the same order |
| `Fee` | Numeric | Exchange fee (proportional to Size USD) |
| `Start Position` | Numeric | Cumulative position after this fill |

> ⚠️ `Closed PnL = 0` for every BUY fill. This is not noise — it is how leveraged trading works. It introduced a structural artifact that made the baseline model completely misleading.

### Fear & Greed Index (`fear_greed_index.csv`)

Daily composite sentiment indicator (0–100) computed from volatility, volume, social media, surveys, Bitcoin dominance, and Google Trends.

| Score | Label | Interpretation |
|---|---|---|
| 0 – 24 | Extreme Fear | Panic selling; contrarian buy signal |
| 25 – 44 | Fear | Bearish sentiment; cautious investors |
| 45 – 55 | Neutral | No strong directional conviction |
| 56 – 74 | Greed | FOMO beginning to drive buying |
| 75 – 100 | Extreme Greed | Market overheated; elevated correction risk |

---

## 🔬 Stage-by-Stage Breakdown

### Stage 1 — Data Preparation & EDA

- Parsed non-standard `DD-MM-YYYY HH:MM` timestamps
- Left-joined fills with Fear & Greed by calendar date
- Engineered: `hour`, `day_of_week`, `is_profitable`
- **Key finding:** Extreme Greed = highest win rate (46.5%) and best avg PnL ($67.89) — this trader uses **momentum**, not contrarian, strategy

| Sentiment | Avg PnL/Fill | Win Rate | Volume |
|---|---|---|---|
| Extreme Fear | $34.54 | 37.1% | $114M |
| Fear | $54.29 | 42.1% | $483M |
| Neutral | $34.31 | 39.7% | $180M |
| Greed | $42.74 | 38.5% | $289M |
| **Extreme Greed** | **$67.89** | **46.5%** | $124M |

*Kruskal-Wallis test: p < 0.05 confirms statistically significant differences across groups.*

---

### Stage 2 — Baseline Model & Leakage Trap

A Random Forest trained on fill-level data with a **random** 80/20 split produced:

> **AUC = 0.9922 — entirely false.**

#### Three compounding leakage errors

| Error | Description |
|---|---|
| **Random split on time-series** | Shuffling rows causes future trades (April 2025) to train on past trades (June 2024). The model sees tomorrow's data. |
| **Execution Price as time proxy** | In a bull market, high price = recent = artificially profitable period. 51.5% feature importance — non-causal. |
| **Mathematical redundancy** | `Size Tokens = Size USD / Execution Price` — removing one leaky feature doesn't help, the information persists via this identity. |

---

### Stage 3 — Leakage Diagnosis & Temporal Split

**Features removed:**

| Feature | Reason |
|---|---|
| `Execution Price` | Bull-run time proxy |
| `Size Tokens` | Reconstructs Execution Price mathematically |
| `Fee` | Post-execution computation; indirect circular leakage |
| `Start Position` | Encodes full history of preceding decisions |
| `log_size_usd` | Deterministic transform — zero new information for trees |
| `fg_cat` | Perfectly collinear with `fg_value` |

**After temporal split (chronological 80/20):**

| Model | CV AUC | Test AUC | Interpretation |
|---|---|---|---|
| Random Forest | 0.658 ± 0.110 | 0.462 | Regime break detected |
| XGBoost | 0.634 ± 0.086 | 0.470 | Same conclusion |

> AUC < 0.5 is not a model failure. April 2025 coincided with a major macroeconomic shock (tariff announcements) and a crypto selloff. The bull-market patterns inverted.

---

### Stage 4 — Trade-Level Reconstruction

**The unit of analysis was wrong.** A fill is not a decision — the order is.

Aggregated 211,224 fills → **27,329 trade-level rows** using `Order ID`.

| Output Column | Aggregation |
|---|---|
| `open_price` | VWAP across all BUY fills |
| `direction` | LONG / SHORT based on BUY/SELL fill counts |
| `total_size_usd` | Sum of all fill notional |
| `net_pnl` | Sum of Closed PnL minus total fees |
| `is_profitable` | 1 if net_pnl > 0 |

**Impact:**

| Metric | Fill-Level | Trade-Level |
|---|---|---|
| Total rows | 211,224 | 27,329 |
| Profitable rate | 41.1% | **24.7%** |

The drop from 41.1% to 24.7% is more honest — BUY fills (always PnL=0) were artificially inflating the fill-level rate.

---

### Stage 5 — Alpha Feature Engineering

13 features capturing price context the model previously had no access to:

| Feature | Category | Description |
|---|---|---|
| `log_size` | Position sizing | log(1 + total_size_usd) — compresses right-skew |
| `direction_enc` | Trade direction | 1 = LONG, 0 = SHORT |
| `fg_value` | Market sentiment | Daily Fear & Greed score (0–100) |
| `hour` | Time context | Hour of execution (0–23) |
| `day_of_week` | Time context | 0 = Monday, 6 = Sunday |
| `mom_1h` | **Price momentum** | % price return in prior 60 minutes |
| `mom_4h` | **Price momentum** | % price return in prior 4 hours |
| `mom_24h` | **Price momentum** | % price return in prior 24 hours |
| `vol_1h` | **Volatility** | Std dev of execution prices — last 60 min |
| `vol_4h` | **Volatility** | Std dev of execution prices — last 4 hours |
| `fee_ratio` | Execution quality | Fee / trade size — maker/taker urgency proxy |
| `rolling_winrate` | Trader performance | Win rate of preceding 20 trades (lagged by 1) |
| `n_fills` | Order complexity | Number of sub-fills |

---

### Stage 6 — Final Model Results

**Temporal CV + Test split (80/20 chronological)**

| Metric | Random Forest | XGBoost |
|---|---|---|
| CV AUC | 0.861 ± 0.039 | **0.872 ± 0.043** |
| **Test AUC (future data)** | 0.877 | **0.910** |
| CV → Test Gap | -0.017 | -0.037 |
| Accuracy | 81.5% | 81.0% |
| F1 Macro | 0.762 | 0.759 |
| Training Time | 5.3s | **0.7s** |

> ✅ **XGBoost selected** — Test AUC = 0.91 on never-seen future trades. The negative CV-to-Test gap (model improves on test) suggests the most recent regime is particularly structured.

#### Key Hyperparameter Choices

```python
XGBClassifier(
    n_estimators=400,
    max_depth=4,           # Shallow — financial signals don't need deep logic
    learning_rate=0.03,    # Slow + conservative, compensated by more trees
    subsample=0.75,        # 75% of rows per tree — prevents co-adaptation
    colsample_bytree=0.75, # 75% of features per tree
    min_child_weight=10,   # Min 10 samples per leaf — avoids statistical accidents
    scale_pos_weight=sp*1.2  # Up-weight the 24.7% minority (profitable) class
)
```

#### Feature Importance

`mom_1h` (short-term momentum) is the dominant signal, confirming this trader's profitability is tightly linked to **short-term trend continuation**.

---

## 🎯 Threshold Analysis

| Threshold | Win Rate (Precision) | Capture Rate (Recall) | F1 | Trades Flagged |
|---|---|---|---|---|
| 0.50 | 52.6% | 84.5% | 0.648 | 1,566 |
| 0.55 | 54.9% | 81.1% | 0.655 | 1,440 |
| 0.60 | 58.2% | 77.6% | 0.665 | 1,300 |
| **0.65 ★** | **61.1%** | **74.3%** | **0.670** | **1,186** |
| 0.70 | 63.7% | 68.5% | 0.660 | 1,047 |
| 0.80 | 73.7% | 52.1% | 0.610 | 688 |
| 0.90 | 84.7% | 26.1% | 0.399 | 300 |

**Baseline (no model):** 24.7% win rate  
**Model at threshold 0.65:** 61.1% win rate — **2.5× improvement**

### Use-Case Guide

| Use Case | Threshold | Win Rate |
|---|---|---|
| Balanced — maximise value | **0.65** | 61.1% |
| Conservative — capital preservation | 0.80 | 73.7% |
| Aggressive — maximise capture | 0.50 | 52.6% |
| Research — highest confidence only | 0.90 | 84.7% |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### Run the Notebook

```bash
git clone https://github.com/RK-NerdyBirdy/<repo-name>.git
cd <repo-name>
jupyter notebook task.ipynb
```

Ensure `historical_data.csv` and `fear_greed_index.csv` are in the working directory before running.

### Inference — Scoring a New Trade

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load('best_model.pkl')

new_trade = pd.DataFrame([{
    'log_size'        : np.log1p(5000),   # $5,000 position
    'direction_enc'   : 1,                 # LONG
    'fg_value'        : 72.0,              # Greed market
    'hour'            : 14,                # 2 PM IST
    'day_of_week'     : 2,                 # Wednesday
    'mom_1h'          : 0.8,               # +0.8% last hour
    'mom_4h'          : 2.1,               # +2.1% last 4h
    'mom_24h'         : 5.3,               # +5.3% last 24h
    'vol_1h'          : 0.05,
    'vol_4h'          : 0.12,
    'fee_ratio'       : 0.0004,
    'rolling_winrate' : 0.35,
    'n_fills'         : 4,
}])

prob = model.predict_proba(new_trade)[0, 1]
THRESHOLD = 0.65
decision = 'TAKE TRADE' if prob >= THRESHOLD else 'SKIP TRADE'

print(f'Profitable probability : {prob:.4f}')
print(f'Decision (t={THRESHOLD})       : {decision}')
```

---

## 💡 Key Findings

1. **The unit of analysis was everything.** Moving from fill-level to trade-level was the single most impactful change. No feature engineering or model tuning can fix fundamentally wrong data framing.

2. **Data leakage inflated AUC from 0.91 to 0.99.** Three subtle, compounding errors produced a publishable-looking but entirely false result. Explicit leakage diagnosis is non-negotiable in financial ML.

3. **Price momentum is the dominant signal.** 1-hour and 4-hour momentum became the top predictors after correct framing. This trader's profitability is tightly linked to short-term trend continuation.

4. **Sentiment matters, but non-linearly.** Fear & Greed carried meaningful signal — but only in combination with price context. Linear correlation (0.01) missed this entirely.

5. **Market regimes change abruptly.** The clean-feature model collapsed from 0.65 CV AUC to 0.47 test AUC because April 2025 broke sharply from 2024 patterns. No static model survives indefinitely without retraining.

---

## 🔭 Future Improvements

- **External OHLCV data** — proper candle data from an exchange API for more precise momentum/volatility features
- **Coin-specific relative strength** — a coin's return relative to Bitcoin for per-coin context
- **Funding rates** — extreme positive funding signals overcrowded longs and elevated reversal risk
- **Liquidation heatmaps** — clusters of leveraged positions create predictable momentum cascades
- **Walk-forward retraining** — retrain monthly on a rolling window to prevent regime staleness
- **Alternative targets (MAE/MFE)** — predict Maximum Adverse/Favorable Excursion for risk/reward optimisation

---

## 📖 Glossary (Selected Terms)

| Term | Definition |
|---|---|
| **AUC** | Area Under the ROC Curve. 0.5 = random, 1.0 = perfect. |
| **Data Leakage** | When information unavailable at prediction time is used during training. Produces inflated metrics that fail in deployment. |
| **Fill** | A single sub-execution of a larger order. |
| **Momentum** | Tendency of trending assets to continue in the same direction. One of the most documented factors in quantitative finance. |
| **Temporal Split** | Dividing time-series by time (not randomly). Training on the past, testing on the future — the only valid protocol for financial ML. |
| **VWAP** | Volume-Weighted Average Price. The true cost basis of an institutional position. |
| **XGBoost** | Extreme Gradient Boosting. Sequential tree-building with L1/L2 regularisation. |

Full glossary of 50+ terms available in the [project report](crypto_overview.docx).

---

## 👤 Author

**Maneet Gupta**  
Ghaziabad, Uttar Pradesh, India  
📧 robomaneet@gmail.com | 📞 +91-7982076022  
🔗 [LinkedIn](https://linkedin.com/in/maneet-gupta) · [GitHub](https://github.com/RK-NerdyBirdy)

---

## 📄 License

This project is licensed under the GNU License. See `LICENSE` for details.

---

<div align="center">
<i>Built with ❤️ using Python, XGBoost, and real on-chain data</i>
</div>
