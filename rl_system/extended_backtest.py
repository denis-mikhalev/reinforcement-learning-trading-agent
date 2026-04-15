import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import MarketTradingEnv

ALGO_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def run_extended_backtest(model_path: Path, config_path: Path, days: int = 120) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    symbol = config["symbol"]
    timeframe = config["timeframe"]
    base_days = config.get("days", 730)
    algo = config["algorithm"]

    if algo not in ALGO_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algo}")

    loader = DataLoader(cache_dir="data/cache")
    # Загружаем больше данных, чем базовый train/test, чтобы было окно >= days
    df = loader.load_data(symbol=symbol, timeframe=timeframe, days=max(base_days, days + 365))

    engineer = FeatureEngineer()
    df = engineer.calculate_features(df)
    if config.get("use_mtf", True):
        df = engineer.add_multi_timeframe_features(df)
    df = engineer.normalize_features(df, method="zscore")

    # Берём последние `days` дней как extended test window
    if days is not None:
        cutoff = df.index.max() - np.timedelta64(days, "D")
        df_test = df[df.index >= cutoff].copy()
    else:
        df_test = df.copy()

    print(f"Extended backtest on {symbol} {timeframe} for last {days} days: {len(df_test)} bars")

    model_cls = ALGO_CLASSES[algo]
    model = model_cls.load(str(model_path))

    env = MarketTradingEnv(
        df=df_test,
        initial_balance=config.get("initial_balance", 10000.0),
        commission=config.get("commission", 0.0006),
        slippage=config.get("slippage", 0.0003),
        lookback_window=config.get("lookback", 64),
        position_size=config.get("position_size", 0.25),
        enable_short=config.get("enable_short", True),
        stop_loss_pct=config.get("stop_loss_pct", config.get("stop_loss", 0.0)),
        max_holding_bars=config.get("max_holding_bars", 0),
    )

    obs, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    info = env._get_info()

    print("\nExtended backtest results:")
    print(json.dumps({
        "episode_reward": float(episode_reward),
        "total_return_pct": float(np.round(info["total_return_pct"], 2)),
        "profit_factor": float(np.round(info["profit_factor"], 2)),
        "total_trades": int(info["total_trades"]),
        "win_rate_pct": float(np.round(info["win_rate"] * 100, 2)),
        "max_drawdown_pct": float(np.round(info["max_drawdown_pct"], 2)),
    }, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended backtest for a trained RL model on a longer window")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model .zip (checkpoint or best/final)")
    parser.add_argument("--config", type=str, required=True, help="Path to base config.json for this model")
    parser.add_argument("--days", type=int, default=120, help="Number of days for extended backtest window (from the end)")
    args = parser.parse_args()

    run_extended_backtest(Path(args.model_path), Path(args.config), days=args.days)
