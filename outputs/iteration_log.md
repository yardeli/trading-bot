# Auto-Trader Iteration Log

## Strategy: Regime-Adaptive Crypto Multi-Strategy
**Domain:** Crypto (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE, LINK, NEAR)
**Models:** XGBoost + LightGBM + Ridge ensemble with recency-weighted training
**Sizing:** Quarter-Kelly with drawdown-adaptive + inverse volatility scaling

| Iter | Change | Return% | Sharpe | WinRate | MaxDD | Trades | Verdict |
|------|--------|---------|--------|---------|-------|--------|---------|
| 0 | Baseline (5 assets) | +414.6% | 1.139 | 43.1% | 20.7% | 276 | Baseline |
| 1 | Trend filter + tighter threshold | +87.8% | 0.358 | 45.6% | 30.2% | 985 | DISCARD |
| 2 | Recency weighting + aggressive Kelly | +866.4% | 0.640 | 45.2% | 21.6% | 217 | KEEP (partial) |
| 3 | Softer drawdown scaling (sqrt + floor) | +920.5% | 1.286 | 42.3% | 32.8% | 260 | KEEP |
| 4 | Volume filter + 15 asset universe | +229.3% | 1.083 | 38.7% | 32.8% | 1183 | KEEP filter |
| 5 | Expanding training window | +473.9% | 0.558 | 46.0% | 35.5% | 1486 | DISCARD |
| 6 | 14-day time exit | +909.1% | 1.347 | 49.0% | 27.3% | 298 | KEEP |
| 7 | Model disagreement penalty | +584.3% | 1.271 | 49.2% | 25.2% | 297 | DISCARD |
| 8 | Tuned XGB/LGB (350 trees, depth 6) | +981.3% | 1.307 | 50.8% | 26.7% | 305 | KEEP |
| CPCV | Validation (15 splits) | 51.4% acc | 1.4% std | — | — | — | VALID |
| 10 | **Inverse vol position sizing** | **+1136.1%** | **1.508** | **50.8%** | **17.2%** | 305 | **BEST** |
| 11 | 5-day target + 10-day hold | +725.9% | 1.377 | 50.3% | 22.5% | 362 | DISCARD |

## Best Configuration (Iteration #10)
- **Total Return: +1,136%** ($10K → $123.6K over 4 years)
- **Sharpe Ratio: 1.508**
- **Max Drawdown: 17.2%**
- **Win Rate: 50.8%**
- **Profit Factor: 1.39**
- **CPCV Validated: 51.4% accuracy, 1.4% std across 15 purged splits**

## Key Findings
1. **Inverse volatility sizing** was the single biggest improvement — cut DD from 27% to 17%
2. **Recency-weighted training** helps the model adapt to recent market regimes
3. **Time-based exits** (14 days) prevent capital lock-up and improve win rate
4. **Softer drawdown scaling** (sqrt curve with 20% floor) keeps trading during drawdowns
5. **More assets ≠ better** — edge concentrates in top-5 liquid majors
6. **Expanding window hurts** — sliding window with recency bias works better
