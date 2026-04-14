# Reinforcement Learning Trading Agent

A deep reinforcement learning system for financial market signal generation, built as a practical exploration of RL applied to real-world sequential decision-making.

The agent learns trading strategies through interaction with a custom Gymnasium environment, using historical OHLCV market data. It supports multiple RL algorithms (PPO, A2C, SAC, TD3), multi-timeframe feature engineering, batch training, model quality assessment, and live signal generation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                          │
│  Market Data ──→ Feature Engineering ──→ Gym Environment │
│  (Binance)       (50+ indicators)       (Custom)         │
│                                              │           │
│                                              ▼           │
│                                         RL Agent         │
│                                    (PPO/A2C/SAC/TD3)     │
│                                              │           │
│                                              ▼           │
│                                    Model Checkpoints     │
│                                              │           │
│                          ┌───────────────────┤           │
│                          ▼                   ▼           │
│                   Quality Assessment    Evaluation        │
│                   (4-level grading)    (Backtesting)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Live Signal Pipeline                     │
│                                                          │
│  Model Registry ──→ Live Agents ──→ Signal Aggregator    │
│  (text file)        (per model)     (ensemble consensus) │
│                                              │           │
│                                              ▼           │
│                                     Telegram / Console   │
└─────────────────────────────────────────────────────────┘
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Trading Environment** | `rl_system/trading_env.py` | Gymnasium-compatible env with long/short positions, commission, slippage, stop-loss |
| **Feature Engineering** | `rl_system/feature_engineering.py` | 50+ technical indicators: trend, momentum, volatility, volume, price action |
| **RL Agent** | `rl_system/rl_agent.py` | Model wrapper for inference, signal generation, backtesting |
| **Data Loader** | `rl_system/data_loader.py` | Market data fetching with caching (Binance API, no keys required) |
| **Training** | `rl_system/train_agent_v2.py` | Full training pipeline with callbacks, early stopping, plateau detection |
| **Batch Training** | `rl_system/batch_train.py` | Sequential training of multiple configurations from a queue file |
| **Quality Assessment** | `rl_system/model_quality_assessment.py` | 4-level model grading: critical → risk → stability → learning |
| **Evaluation** | `rl_system/evaluate_agent.py` | Backtesting with equity curves, action distribution, trade analysis |
| **Live Inference** | `rl_system/run_live_from_registry.py` | Multi-model live signal generation from model registry |
| **Ensemble** | `rl_system/live_signals_summary.py` | Signal aggregation across models with consensus voting |
| **Config Manager** | `rl_system/config_manager.py` | Presets (conservative/balanced/aggressive) and per-algorithm defaults |

## Tech Stack

- **RL Framework**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO, A2C, SAC, TD3)
- **Environment**: [Gymnasium](https://gymnasium.farama.org/)
- **Deep Learning**: PyTorch
- **Feature Engineering**: TA-Lib, pandas, numpy
- **Data Source**: Binance public API (via ccxt / python-binance)
- **Notifications**: Telegram Bot API (optional)

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configure (Optional)

Copy `.env.example` to `.env` and fill in Telegram credentials for notifications:

```bash
cp .env.example .env
```

### 3. Train a Model

```bash
python rl_system/train_agent_v2.py \
    --symbol BTCUSDT \
    --timeframe 4h \
    --algorithm PPO \
    --days 1500 \
    --total-timesteps 500000 \
    --enable-short
```

### 4. Evaluate

```bash
python rl_system/evaluate_agent.py \
    --model-path rl_system/models/BTCUSDT_4h_PPO_... \
    --days 180 \
    --deterministic \
    --save-results
```

### 5. Live Signals

Add models to `rl_system/live_models_registry.txt`:

```
BTCUSDT_4h_PPO_1500d_bt180d_20260201    rl_model_500000_steps
```

Then run:

```bash
python rl_system/run_live_from_registry.py
```

## Model Quality Assessment

The system includes a 4-level quality gate for trained models:

| Level | Checks | Purpose |
|-------|--------|---------|
| **Critical** | Min return, min profit factor, max drawdown | Basic viability |
| **Risk** | Drawdown ratios, Sharpe, return/DD ratio | Risk management |
| **Stability** | Profitable zones, overfitting detection, consistency | Robustness |
| **Learning** | Plateau detection, convergence analysis | Training quality |

Verdicts: ✅ READY FOR LIVE · ⚠️ USE WITH CAUTION · 📄 PAPER TRADING ONLY · ❌ NOT READY

## Project Structure

```
.
├── rl_system/
│   ├── trading_env.py          # Gymnasium trading environment
│   ├── rl_agent.py             # Agent wrapper
│   ├── feature_engineering.py  # Technical indicators (50+)
│   ├── data_loader.py          # Market data with caching
│   ├── config_manager.py       # Configuration management
│   ├── train_agent_v2.py       # Training pipeline
│   ├── batch_train.py          # Batch training from queue
│   ├── evaluate_agent.py       # Backtesting & analysis
│   ├── evaluate_stability.py   # Multi-period stability
│   ├── model_quality_assessment.py  # 4-level quality gate
│   ├── select_best_model.py    # Checkpoint selection
│   ├── run_live_from_registry.py    # Live multi-model inference
│   ├── live_signals_summary.py      # Ensemble signal aggregation
│   └── configs/                # Algorithm & symbol configs
├── configs/                    # Model architecture configs
├── docs/                       # Guides and documentation
├── telegram_sender.py          # Telegram notifications
├── risk_metrics.py             # Risk calculation utilities
├── requirements.txt
└── .env.example
```

## Configuration

Training configs are JSON files supporting:

- **Algorithm selection**: PPO, A2C, SAC, TD3 with per-algorithm hyperparameters
- **Environment params**: initial balance, commission, slippage, lookback window
- **Policy params**: position sizing, short selling, stop-loss
- **Training params**: learning rate, batch size, gamma, clip range, total timesteps

See `configs/V3.2e.json` for an example configuration.

## Documentation

- [RL Training Guide](docs/RL_TRAINING_GUIDE.md) — Training workflow and best practices
- [Models Evaluation](docs/RL_MODELS_EVALUATION.md) — Evaluation methodology and metrics
- [GPU Optimization Guide](docs/GPU_OPTIMIZATION_GUIDE.md) — GPU setup for training acceleration
- [PPO Parameters Guide](rl_system/PPO_PARAMETERS_GUIDE.md) — PPO hyperparameter tuning reference

## License

MIT
