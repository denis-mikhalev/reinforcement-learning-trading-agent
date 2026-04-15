# System Capabilities

A complete overview of all features and tools available in this RL trading agent system.

---

## 🧠 Reinforcement Learning Core

### Supported Algorithms

| Algorithm | Type | Key Strength |
|-----------|------|-------------|
| **PPO** (Proximal Policy Optimization) | On-policy | Stable training, clip-based updates |
| **A2C** (Advantage Actor-Critic) | On-policy | Fast convergence, good baseline |
| **SAC** (Soft Actor-Critic) | Off-policy | Entropy-regularized exploration |
| **TD3** (Twin Delayed DDPG) | Off-policy | Deterministic policy, reduced overestimation |

### Custom Gymnasium Environment

The trading environment (`CryptoTradingEnv`) simulates realistic market conditions:

- **3 actions**: BUY, SELL, HOLD
- **3 position states**: Long (+1), Flat (0), Short (-1)
- **Realistic friction**: configurable commission (default 0.1%), slippage (default 0.05%)
- **Risk controls**: stop-loss (% of balance), max holding bars (force exit)
- **Observation space**: `[lookback_window × n_features]` tensor with position metadata
- **30+ tracked metrics**: equity curve, PnL, drawdown, win rate, profit factor, trade duration, and more

### Reward Functions

Three reward variants for different optimization objectives:

| Variant | Strategy | Use Case |
|---------|----------|----------|
| **Default (v5)** | Balanced PnL + shaping | General purpose |
| **Variant 2** | High HOLD reward (×100 unrealized PnL multiplier) | Encourages holding profitable positions longer |
| **Variant 3** | Realized PnL only (zero intermediate rewards) | Pure trade-closure optimization |

---

## 📊 Feature Engineering — 99 Indicators

Nine categories of technical features computed from OHLCV data:

| # | Category | Examples |
|---|----------|----------|
| 1 | **Price Action** | Returns (7/14/21/50 bar), HL range, OC range, wick ratios, body size |
| 2 | **Trend** | SMA, EMA (7/14/21/50), MACD (line, signal, histogram), ADX |
| 3 | **Momentum** | RSI (7/14/21), Stochastic (%K, %D), ROC |
| 4 | **Volatility** | ATR, Bollinger Bands (%B, upper/lower, bandwidth) |
| 5 | **Volume** | OBV, Volume SMA, Volume ROC |
| 6 | **Market Structure** | Candle patterns, structural levels |
| 7 | **Volume Profile** | Volume distribution analysis |
| 8 | **Order Flow Proxies** | Buy/sell pressure estimation |
| 9 | **Market Regime** | Trending vs ranging detection |

**Processing pipeline**: Z-score normalization → 0.1% extreme clipping → NaN/Inf handling

### Multi-Timeframe Features (Optional)

Aggregates weekly and monthly candles onto the base timeframe, adding 15–20 macro-level features for multi-scale trend analysis.

### Multi-Head Attention Extractor (Optional)

Custom PyTorch feature extractor using Transformer-style attention:
- 4 attention heads, 2 layers (configurable)
- Residual connections + LayerNorm
- Learns to focus on important price history regions

---

## 🚀 Training Pipeline

### Single Model Training

```bash
python rl_system/train_agent_v2.py \
    --symbol BTCUSDT --timeframe 4h --algorithm PPO \
    --days 1500 --total-timesteps 500000 \
    --enable-short --use-mtf --telegram
```

**Key training parameters:**

| Parameter | Description |
|-----------|-------------|
| `--symbol` | Trading pair (BTCUSDT, ADAUSDT, XRPUSDT, etc.) |
| `--timeframe` | Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d) |
| `--algorithm` | PPO, A2C, SAC, or TD3 |
| `--days` | Historical data period |
| `--total-timesteps` | Training steps |
| `--lookback` | Observation window size (bars) |
| `--position-size` | Capital allocation per trade (0.0–1.0) |
| `--enable-short` | Allow short positions |
| `--stop-loss` | Stop-loss as % of balance |
| `--use-mtf` | Enable multi-timeframe features |
| `--use-attention` | Enable attention feature extractor |
| `--config` | Load all params from a JSON file |
| `--telegram` | Send completion notification to Telegram |

**Training features:**
- Automatic 80/20 train/test split
- Periodic checkpoint saving with evaluation
- Early stopping on eval reward / Sharpe ratio
- Plateau detection during training
- 6-panel checkpoint comparison visualization (auto-generated PNG)
- Telegram notification on completion with full metrics

### Batch Training

```bash
python rl_system/batch_train.py \
    --config-list rl_system/training_queue.txt \
    --continue-on-error
```

Trains multiple configurations sequentially from a queue file — ideal for overnight runs.

### Model Output Structure

```
rl_system/models/BTCUSDT_4h_A2C_1095d_bt217d_20251127/
├── config.json                    # Full training configuration
├── final_model.zip                # Final model weights
├── best/best_model.zip            # Best checkpoint by eval metric
├── checkpoints/rl_model_*.zip     # Periodic checkpoints
├── vec_normalize.pkl              # Observation normalization stats
├── CHECKPOINTS_COMPARISON.md      # Plateau analysis report
├── CHECKPOINTS_COMPARISON.png     # 6-panel visualization
└── eval_logs/                     # Evaluation traces
```

---

## ✅ Model Quality Assessment

### 4-Level Quality Gate

Every trained model is evaluated through a rigorous multi-level assessment:

| Level | Category | What It Checks |
|-------|----------|---------------|
| 1 | **Critical** | Minimum return > 0%, profit factor ≥ 1.0, max drawdown ≤ 30%, min 5 trades |
| 2 | **Risk** | Sharpe ratio, return/drawdown ratio, drawdown severity |
| 3 | **Stability** | Profitable zone coverage, overfitting detection (best vs final gap), tail consistency |
| 4 | **Learning** | Training progression, convergence, no late-stage collapse |

**Verdicts:**
- ✅ **READY FOR LIVE** — passes all levels
- ⚠️ **USE WITH CAUTION** — minor issues detected
- 📄 **PAPER TRADING ONLY** — needs more validation
- ❌ **NOT READY** — fails critical checks

### Evaluation Tools

| Tool | Command | Purpose |
|------|---------|---------|
| Quick Eval | `python rl_system/quick_eval_model.py <model_path>` | Fast backtest: return, PF, trades, win rate |
| Extended Backtest | `python rl_system/extended_backtest.py --model-path <path> --days 120` | Stress test on recent data |
| Walk-Forward | `python rl_system/evaluate_stability.py --model <path> --days 180` | Multi-period stability analysis |
| Full Analysis | `python rl_system/evaluate_agent.py --model-path <path> --save-results` | Equity curves, trade breakdown, action distribution |
| Best Checkpoint | `python rl_system/select_best_model.py <model_dir>` | Finds optimal checkpoint by composite score |
| Plateau Analysis | `python rl_system/plateau_analysis.py` | Detects training plateau regions |

---

## 📡 Live Signal Generation

### Single Model

```bash
python rl_system/run_live_agent.py \
    --model-path rl_system/models/BTCUSDT_4h_A2C_... \
    --continuous --telegram --signal-change-sound
```

Real-time inference with candle polling, Telegram alerts, and optional sound notifications.

### Multi-Model (Registry)

```bash
# rl_system/live_models_registry.txt
ADAUSDT_4h_A2C_1300d_bt60d_20251212  rl_model_280000_steps
BTCUSDT_4h_PPO_2500d_bt500d_20251215  rl_model_450000_steps
```

```bash
python rl_system/run_live_from_registry.py
```

Launches all registered models in parallel with automatic stale state cleanup.

### Ensemble Signal Aggregation

```bash
python rl_system/live_signals_summary.py \
    --interval 60 --telegram --candle-close-sound
```

- Groups signals by symbol and action (BUY / SELL / HOLD)
- **Majority voting** consensus across models
- Stuck model detection (flags single signal > 7 days)
- Signal change notifications
- Formatted Telegram table output

---

## 📈 Analysis & Comparison

| Tool | Purpose |
|------|---------|
| `compare_all_models.py` | Full comparison table of all trained models |
| `quick_compare_models.py` | Lightweight comparison (key metrics only) |
| `analyze_stuck_models.py` | Detect models stuck on one signal for too long |
| `analyze_trades_csv.py` | Detailed trade-by-trade analysis from CSV |
| `generate_models_md_table.py` | Auto-generate markdown performance tables |
| `regenerate_checkpoint_visualization.py` | Regenerate training progress charts |

---

## ⚙️ Configuration System

### 3 Built-in Presets

| Preset | Position Size | Shorts | Learning Rate | Clip Range | Risk Level |
|--------|:------------:|:------:|:-------------:|:----------:|:----------:|
| **Conservative** | 80% | ❌ | 1e-4 | 0.1 | Low |
| **Balanced** | 95% | ✅ | 3e-4 | 0.2 | Medium |
| **Aggressive** | 100% | ✅ | 5e-4 | 0.3 | High |

### Per-Algorithm Defaults

Each algorithm (PPO, A2C, SAC, TD3) has its own default configuration with tuned hyperparameters — see `rl_system/configs/`.

### JSON Training Configs

Full training runs are reproducible via JSON configs:

```bash
python rl_system/train_agent_v2.py --config configs/V3.2e.json
```

Configs specify everything: algorithm, symbol, timeframe, hyperparameters, environment settings, and feature flags.

---

## 🛡️ Risk Management

### Environment-Level

| Parameter | Default | Description |
|-----------|---------|-------------|
| Commission | 0.1% | Per-trade fee |
| Slippage | 0.05% | Directional price impact |
| Stop-Loss | Configurable | % of initial balance |
| Max Holding | Configurable | Force exit after N bars |
| Position Size | 95% | Capital allocation per trade |

### Signal-Level Risk Metrics

The `risk_metrics.py` module computes per-signal risk assessment:

| Metric | Description |
|--------|-------------|
| **RRR** | Risk-Reward Ratio (target / stop distance) |
| **P_BE** | Break-even probability |
| **EV** | Expected value (%) |
| **Net Target** | Target % minus round-trip costs |
| **Edge Filter** | Minimum net move threshold |

Symbol-specific cost overrides via `risk_config.json`.

---

## 🔔 Monitoring & Notifications

- **Telegram Bot**: Training completion alerts, live signal updates, ensemble summaries
- **Sound Alerts**: Per-signal or signal-change-only audio notifications
- **Stuck Model Detection**: Automatic flagging of models producing the same signal for extended periods
- **Live State Files**: JSON state output per model for ensemble aggregation

---

## 🗄️ Data Management

| Command | Purpose |
|---------|---------|
| `python rl_system/download_data.py --symbol BTCUSDT --timeframe 4h --days 1500` | Pre-download and cache market data |
| `python rl_system/clear_data_cache.py --list` | View cached datasets |
| `python rl_system/clear_data_cache.py --symbol BTCUSDT` | Clear specific cache |
| `python rl_system/cleanup_unused_checkpoints.py --dry-run` | Preview checkpoint cleanup |
| `python rl_system/cleanup_unused_checkpoints.py --yes` | Remove unused checkpoints |

**Data pipeline**: Binance public API → Pickle cache (24h TTL) → Feature engineering → Gymnasium environment

No API keys required — uses public OHLCV endpoints with automatic Binance.US fallback for geo-restricted regions.

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-----------|
| RL Framework | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) |
| Environment | [Gymnasium](https://gymnasium.farama.org/) |
| Deep Learning | PyTorch |
| Indicators | TA-Lib |
| Data Processing | pandas, NumPy |
| Data Source | Binance API (python-binance) |
| Notifications | Telegram Bot API |
| Visualization | Matplotlib |
| Monitoring | TensorBoard |
