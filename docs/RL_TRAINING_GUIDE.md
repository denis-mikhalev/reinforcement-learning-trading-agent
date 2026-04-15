# RL Training Guide (A2C / PPO, `train_agent_v2.py`)

## 1. How models are trained now

Models are trained via `rl_system/train_agent_v2.py` using CLI args and/or JSON configs.
For each run the script:
- Loads OHLCV data via `DataLoader`.
- Builds features via `FeatureEngineer` (enhanced + optional MTF).
- Splits data into train/test using `days` + `train_ratio` or `test_days`/`test_ratio`.
- Creates `MarketTradingEnv` with `lookback`, risk parameters, etc.
- Trains selected algorithm (A2C, PPO, SAC, TD3).
- Saves models and configs into `rl_system/models/<MODEL_NAME>`.

The training config for each trained model is saved as `rl_system/models/<MODEL_NAME>/config.json`.
This file is the single source of truth for how that model was actually trained (symbol, timeframe, days, lookback, risk, hyperparams, etc.).

For the model:
`rl_system/models/BTCUSDT_4h_A2C_1095d_bt217d_20251127_125823`

The `config.json` shows it was trained with:
- `algorithm`: `A2C`
- `symbol`: `BTCUSDT`
- `timeframe`: `4h`
- `days`: `1095`
- `lookback`: `64`
- `position_size`: `0.2`
- `commission`: `0.0006`
- `slippage`: `0.0003`
- `enable_short`: `true`
- `stop_loss_pct`: `0.01`
- `max_holding_bars`: `0`
- `total_timesteps`: `500000`
- `n_features`: `113` (enhanced + MTF)

So this model **was** trained via the new `train_agent_v2.py` with extended features and MTF.


## 2. Recommended way to train a new RL model

### 2.1. Basic A2C training from CLI

From repo root:

```powershell
.\.venv\Scripts\python.exe rl_system\train_agent_v2.py `
    --symbol BTCUSDT `
    --timeframe 4h `
    --days 1095 `
    --algorithm A2C `
    --lookback 64 `
    --position-size 0.2 `
    --enable-short `
    --stop-loss 0.01 `
    --commission 0.0006 `
    --slippage 0.0003 `
    --total-timesteps 500000 `
    --use-mtf `
    --seed 42
```

This will:
- Fetch 1095 days of BTCUSDT 4h data.
- Build enhanced + MTF features.
- Split into train/test with default `train_ratio`.
- Train A2C for 500k steps.
- Save everything under `rl_system/models/<AUTO_MODEL_NAME>`.


### 2.2. Training from JSON config

You can store stable experiment configs in `configs/` and pass them with `--config`.
Example:

```powershell
.\.venv\Scripts\python.exe rl_system\train_agent_v2.py --config configs\my_a2c_4h_1095d.json
```

Minimal example of a config file:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "days": 1095,
  "algorithm": "A2C",
  "lookback": 64,
  "position_size": 0.2,
  "enable_short": true,
  "commission": 0.0006,
  "slippage": 0.0003,
  "stop_loss": 0.01,
  "total_timesteps": 500000,
  "use_mtf": true,
  "seed": 42
}
```

Any field from `parse_args()` can be placed into such JSON; `train_agent_v2.py` will merge it with CLI defaults and save effective config into `rl_system/models/<MODEL_NAME>/config.json`.


### 2.3. Using short test window (`test_days` / `test_ratio`)

To train with a long history and a short fresh test window:

```powershell
.\.venv\Scripts\python.exe rl_system\train_agent_v2.py `
    --symbol BTCUSDT `
    --timeframe 4h `
    --days 1095 `
    --algorithm A2C `
    --lookback 64 `
    --position-size 0.2 `
    --enable-short `
    --stop-loss 0.01 `
    --commission 0.0006 `
    --slippage 0.0003 `
    --total-timesteps 500000 `
    --use-mtf `
    --test-days 90
```

Priority of splitting:
- If `test_days` set and > 0 → use last `test_days` days as test.
- Else if `test_ratio` set → use that fraction for test.
- Else → use `train_ratio`.


### 2.4. Naming and locating trained models

- Every run creates `rl_system/models/<MODEL_NAME>` where:
  - `<MODEL_NAME>` is auto-generated if `--model-name` is not provided:  
    `SYMBOL_TIMEFRAME_ALGO_<DAYS>d_bt<TEST_DAYS>d_<TIMESTAMP>`
- Inside the folder you will find:
  - `config.json` — full effective training config + evaluation metrics.
  - `final_model.zip` — final checkpoint after `total_timesteps`.
  - `best/best_model.zip` + `best/config.json` — best eval checkpoint.
  - `checkpoints/rl_model_*.zip` — periodic checkpoints.
  - `eval_logs/`, `logs/` — training and eval logs.

For live trading we typically:
- Run `rl_system/select_best_model.py` on this folder to pick the best checkpoint by trading metrics.
- Write chosen checkpoint path into `rl_system/best_models_summary.json` (field `model_path`).


## 3. Checklist when training a model for live use

1. **Define experiment config**
   - Symbol, timeframe, days.
   - Algorithm (A2C/PPO) and hyperparams.
   - Risk: `position_size`, `stop_loss_pct`, `enable_short`.
   - `lookback` window.
   - `use_mtf` (true for current A2C models).
2. **Run training** via `train_agent_v2.py` (CLI or `--config`).
3. **Inspect `config.json`** in `rl_system/models/<MODEL_NAME>`:
   - `algorithm`, `lookback`, `n_features`, `train_period`, `test_period`.
4. **Evaluate and select checkpoint** with `select_best_model.py`.
5. **Register best checkpoint** in `rl_system/best_models_summary.json`:
   - `label`, `description`, `model_path` to specific `.zip`, `metrics`, `enabled_live`.
6. **Run live** using task `RL Live: from registry`.


## 4. Summary for the current 4h A2C model

- Trained by `rl_system/train_agent_v2.py` with A2C on BTCUSDT 4h, 1095 days.
- `lookback = 64`, `n_features = 113` (enhanced + MTF).
- `model_path` in `best_models_summary.json` should point to the exact checkpoint you want to use (e.g. `checkpoints/rl_model_260000_steps.zip`).
- Live launcher `rl_system/run_live_from_registry.py` + `run_live_agent.py` now expects and reproduces the same env parameters from `config.json`.
