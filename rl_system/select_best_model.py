import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import CryptoTradingEnv
from config_manager import ConfigManager
from plateau_analysis import compute_plateau, compute_live_verdict, load_thresholds_from_config


ALGO_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def calculate_min_trades_dynamic(test_days: int) -> int:
    """Рассчитывает минимальное количество сделок для валидации чекпоинта.
    
    Универсальная формула, НЕ зависящая от таймфрейма:
        min_trades = max(STATISTICAL_MINIMUM, months × TRADES_PER_MONTH)
    
    Обоснование:
        Статистическая значимость зависит от количества наблюдений (сделок),
        а не от таймфрейма. 50 сделок на 4h и 50 сделок на 8h имеют
        одинаковую статистическую надежность.
    
    Args:
        test_days: Длина бэктеста в днях
    
    Returns:
        int: Минимальное количество сделок
    
    Примеры (с настройками по умолчанию STATISTICAL_MINIMUM=30, TRADES_PER_MONTH=10):
        30 дней  → max(30, 1×10)  = 30
        60 дней  → max(30, 2×10)  = 30
        90 дней  → max(30, 3×10)  = 30
        120 дней → max(30, 4×10)  = 40
        180 дней → max(30, 6×10)  = 60
        365 дней → max(30, 12×10) = 120
    
    Настройки можно изменить в ConfigManager.MIN_TRADES_SETTINGS
    """
    settings = ConfigManager.MIN_TRADES_SETTINGS
    statistical_minimum = settings['STATISTICAL_MINIMUM']
    trades_per_month = settings['TRADES_PER_MONTH']
    
    months = test_days / 30.0
    expected_trades = int(months * trades_per_month)
    
    min_trades = max(statistical_minimum, expected_trades)
    
    return min_trades


def load_base_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def build_test_env(df, config: dict) -> CryptoTradingEnv:
    return CryptoTradingEnv(
        df=df,
        initial_balance=config.get("initial_balance", 10000.0),
        commission=config.get("commission", 0.0006),
        slippage=config.get("slippage", 0.0003),
        lookback_window=config.get("lookback", 64),
        position_size=config.get("position_size", 0.25),
        enable_short=config.get("enable_short", True),
        stop_loss_pct=config.get("stop_loss_pct", config.get("stop_loss", 0.0)),
        max_holding_bars=config.get("max_holding_bars", 0),
    )


def evaluate_model(model_path: Path, algo: str, df_test, config: dict, trades_log_dir: Path | None = None) -> dict:
    algo_cls = ALGO_CLASSES[algo]
    model = algo_cls.load(str(model_path))

    env = build_test_env(df_test, config)
    obs, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    info = env._get_info()

    # Сохраняем детальный лог сделок для этого конкретного бэктеста, если запрошено
    if trades_log_dir is not None and getattr(env, "trades_log", None):
        try:
            trades_log_dir.mkdir(parents=True, exist_ok=True)
            trades = env.trades_log
            csv_path = trades_log_dir / f"{model_path.stem}_trades.csv"
            fieldnames = list(trades[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
        except Exception as e:
            print(f"⚠️  Failed to save trades log for {model_path.name}: {e}")

    return {
        "model_path": str(model_path),
        "episode_reward": float(episode_reward),
        "total_return_pct": float(np.round(info["total_return_pct"], 2)),
        "profit_factor": float(np.round(info["profit_factor"], 2)),
        "total_trades": int(info["total_trades"]),
        "win_rate_pct": float(np.round(info["win_rate"] * 100, 2)),
        "max_drawdown_pct": float(np.round(info["max_drawdown_pct"], 2)),
        "sharpe_ratio": float(np.round(info.get("sharpe_ratio", 0), 2)),
    }


def extract_timestep_from_checkpoint(checkpoint_path: Path) -> int:
    """Извлекает номер шага обучения из имени чекпоинта.
    
    Примеры:
    - rl_model_320000_steps.zip -> 320000
    - best_model.zip -> 0 (для best/final используем 0 как специальное значение)
    - final_model.zip -> 999999999 (для сортировки в конец)
    """
    name = checkpoint_path.stem
    
    # Для best_model возвращаем максимальное значение для правильной сортировки
    if 'best_model' in name:
        return 999999999
    
    # Для final_model тоже максимальное
    if 'final_model' in name:
        return 999999998
    
    # Для обычных чекпоинтов извлекаем число из имени
    # Формат: rl_model_<steps>_steps.zip
    import re
    match = re.search(r'(\d+)_steps', name)
    if match:
        return int(match.group(1))
    
    return 0


def score_metrics(m: dict, min_trades: int = 30, train_size: int = None) -> float:
    """Скалярный скор для сравнения моделей.

    Базовая логика:
    - отбрасываем модели с количеством сделок < min_trades (скор = -inf)
    - отбрасываем модели с < 5 эпизодов (слишком ранний чекпоинт, ненадёжно)
    - основной приоритет: total_return_pct
    - второй: profit_factor
    - мягкое поощрение большего числа сделок
    - штраф за большую просадку
    - линейный штраф за сверхранние чекпоинты (5-15 эпизодов)
    - бонус за зрелые модели (15-100 эпизодов)
    - мягкий штраф за возможное переобучение (>150 эпизодов)
    """
    if m["total_trades"] < min_trades:
        return -1e9
    
    # Отбрасываем сверхранние чекпоинты (< 5 эпизодов) — слишком мало обучения,
    # результат скорее всего случаен (selection bias)
    episodes = m.get("episodes", 0)
    if episodes < 5.0:
        return -1e9

    score = m["total_return_pct"]
    score += (m["profit_factor"] - 1.0) * 10.0
    score += min(m["total_trades"], 300) * 0.01
    score -= max(m["max_drawdown_pct"], 0) * 0.2
    
    # Линейный штраф для ранних чекпоинтов (5-15 эпизодов):
    # на 5 эпизодах штраф -5.0, на 15 — 0, плавное нарастание доверия
    if episodes < 15.0:
        score -= (15.0 - episodes) * 0.5
    # Бонус за зрелые модели (15-100 эпизодов) — агент достаточно обучился
    elif 15.0 <= episodes <= 100.0:
        score += 2.0
    # Нейтральная зона (100-150) — всё ещё нормально
    elif 100.0 < episodes <= 150.0:
        score += 1.0
    # Мягкий штраф за возможное переобучение (>150 эпизодов)
    elif episodes > 150.0:
        score -= (episodes - 150.0) * 0.03
    return score


def select_best_checkpoint(model_dir: Path, min_trades: int = None, train_size: int = None) -> list:
    """
    Выбирает лучший чекпоинт по торговым метрикам.
    
    Args:
        model_dir: Путь к директории модели
        min_trades: Минимальное количество сделок. Если None - рассчитывается динамически
                   из timeframe и длины бэктеста (рекомендуется).
        train_size: Размер обучающей выборки (количество баров) для расчета эпизодов.
                   Если None - берется из base_config.
    
    Returns:
        list: Список метрик всех протестированных чекпоинтов
    """
    base_config = load_base_config(model_dir)

    symbol = base_config["symbol"]
    timeframe = base_config["timeframe"]
    days = base_config.get("days", 730)
    algo = base_config["algorithm"]

    if algo not in ALGO_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algo}")

    loader = DataLoader(cache_dir="data/cache")
    df_raw = loader.load_data(symbol=symbol, timeframe=timeframe, days=days)

    import pandas as pd

    # Приводим сырые данные к DatetimeIndex по timestamp
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw = df_raw.set_index("timestamp")

    engineer = FeatureEngineer()
    df = engineer.calculate_features(df_raw.copy())

    # После feature engineering длина df может быть меньше из-за lookback/rolling.
    # Восстанавливаем временной индекс аккуратно:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        # Подстраиваемся под хвост исходного индекса, чтобы сохранить временную ось
        if len(df_raw) >= len(df):
            df.index = df_raw.index[-len(df):]
        else:
            raise ValueError("Feature matrix has more rows than raw data; cannot align index correctly.")

    if base_config.get("use_mtf", True):
        df = engineer.add_multi_timeframe_features(df)
    df = engineer.normalize_features(df, method="zscore")

    # Разбиение только по train_ratio (как в обучении): первые X% баров train, хвост test.
    train_ratio = float(base_config.get("train_ratio", 0.8))
    train_ratio = max(0.0, min(0.99, train_ratio))
    split_idx = int(len(df) * train_ratio)
    df_test = df.iloc[split_idx:].copy()

    # Логируем фактический тестовый период по исходным временным меткам
    test_days_actual = 0
    if len(df_test) > 0:
        # df_raw и df уже синхронизированы по времени через engineer; индекса может быть RangeIndex,
        # поэтому для дат опираемся на df_raw.timestamp
        idx_raw_split = int(len(df_raw) * train_ratio)
        test_raw = df_raw.iloc[idx_raw_split:]
        start_ts = test_raw.index[0]
        end_ts = test_raw.index[-1]
        test_days_actual = (end_ts - start_ts).days
        print("🗓  Selector test period (by train_ratio):")
        print(f"   Train ratio: {train_ratio:.2f}  → Test bars: {len(df_test)}")
        print(f"   Test timestamps: {start_ts.strftime('%Y-%m-%d %H:%M:%S')} → {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Test duration: {test_days_actual} days")
    else:
        print("⚠️  Selector test period is empty after train_ratio split – check data length.")
    
    # Динамический расчет min_trades если не указано явно
    if min_trades is None:
        test_days_for_calc = test_days_actual if test_days_actual > 0 else int(days * (1 - train_ratio))
        
        min_trades = calculate_min_trades_dynamic(test_days=test_days_for_calc)
        
        settings = ConfigManager.MIN_TRADES_SETTINGS
        months = test_days_for_calc / 30.0
        expected = int(months * settings['TRADES_PER_MONTH'])
        
        print(f"\n📊 Min trades calculation (universal formula):")
        print(f"   Test period: {test_days_for_calc} days ({months:.1f} months)")
        print(f"   Formula: max({settings['STATISTICAL_MINIMUM']}, {months:.1f} × {settings['TRADES_PER_MONTH']}) = max({settings['STATISTICAL_MINIMUM']}, {expected})")
        print(f"   ✅ Result: min_trades = {min_trades}")
    else:
        print(f"\n📊 Using manual min_trades = {min_trades}")

    # Кандидаты: все checkpoint-zip'ы + final_model + best_model (если есть)
    candidates = []

    checkpoints_dir = model_dir / "checkpoints"
    if checkpoints_dir.exists():
        for p in checkpoints_dir.glob("*.zip"):
            candidates.append(p)

    final_path = model_dir / "final_model.zip"
    if final_path.exists():
        candidates.append(final_path)

    best_path = model_dir / "best" / "best_model.zip"
    if best_path.exists():
        candidates.append(best_path)

    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found in {model_dir}")

    print(f"Found {len(candidates)} candidate models. Evaluating on test set...")

    # Получаем размер обучающей выборки для расчета эпизодов
    if train_size is None:
        train_size = base_config.get("train_size", 0)
        if train_size == 0:
            print("⚠️  Warning: train_size not found in config, episode calculation will be inaccurate")
            train_size = 10000  # fallback значение

    metrics = []
    regular_checkpoints = []  # Только регулярные чекпоинты
    special_models = []  # best_model.zip и final_model.zip

    # Папка для логов сделок всех кандидатов
    trades_logs_dir = model_dir / "checkpoint_trades_logs"
    for p in sorted(candidates):
        m = evaluate_model(p, algo, df_test, base_config, trades_log_dir=trades_logs_dir)
        m["checkpoint"] = p.name  # Добавляем имя чекпоинта
        m["timestep"] = extract_timestep_from_checkpoint(p)  # Добавляем номер шага
        
        # Рассчитываем количество эпизодов
        timestep = m["timestep"]
        if timestep < 900000000:  # Не для special моделей
            m["episodes"] = round(timestep / train_size, 2) if train_size > 0 else 0
        else:
            m["episodes"] = 0  # Для special моделей не считаем
        
        m["score"] = score_metrics(m, min_trades=min_trades, train_size=train_size)
        metrics.append(m)
        
        # Разделяем на регулярные чекпоинты и специальные модели
        if "best_model" in p.name or "final_model" in p.name:
            special_models.append(m)
            marker = "📌" if "best" in p.name else "🏁"
            print(
                f"  {marker} {p.name}: return={m['total_return_pct']:+.2f}% PF={m['profit_factor']:.2f} "
                f"trades={m['total_trades']} DD={m['max_drawdown_pct']:.2f}% score={m['score']:.2f}"
            )
        else:
            regular_checkpoints.append(m)
            episodes = m.get('episodes', 0)
            print(
                f"  {p.name}: ep={episodes:.1f} return={m['total_return_pct']:+.2f}% PF={m['profit_factor']:.2f} "
                f"trades={m['total_trades']} DD={m['max_drawdown_pct']:.2f}% score={m['score']:.2f}"
            )

    # Выбираем лучший ТОЛЬКО среди регулярных чекпоинтов
    if not regular_checkpoints:
        print("\n⚠️  No regular checkpoints found, using all candidates")
        best = max(metrics, key=lambda x: x["score"])
    else:
        best = max(regular_checkpoints, key=lambda x: x["score"])
        print(f"\n✅ Best checkpoint selected from {len(regular_checkpoints)} regular checkpoints")
        if special_models:
            print(f"📌 {len(special_models)} special models (best/final) evaluated for comparison")

    print("\nBest checkpoint by trading metrics (regular checkpoints only):")
    print(json.dumps(best, indent=4))

    # --- Вычисляем агрегатные статистики по чекпоинтам ---
    checkpoint_stats = _compute_checkpoint_stats(
        regular_checkpoints=regular_checkpoints,
        metrics_list=metrics,
        best_checkpoint=best,
        model_dir=model_dir,
        min_trades=min_trades,
    )
    print(f"\n📊 Checkpoint statistics:")
    print(f"   Total checkpoints: {checkpoint_stats['total_checkpoints']}")
    print(f"   Valid checkpoints: {checkpoint_stats['valid_checkpoints']}")
    print(f"   Positive return: {checkpoint_stats['positive_pct']:.0f}%")
    print(f"   Plateau: {'found (' + str(checkpoint_stats['plateau_len']) + ' checkpoints)' if checkpoint_stats['plateau_found'] else 'not found'}")
    print(f"   Tail profitable: {checkpoint_stats['tail_profitable_pct']:.0f}%")
    print(f"   Verdict: {checkpoint_stats['verdict']}")
    if checkpoint_stats['verdict_reasons']:
        for reason in checkpoint_stats['verdict_reasons']:
            print(f"     ❌ {reason}")

    # Сохранить результаты выбора
    selector_config_path = model_dir / "selected_best_by_metrics.json"
    result = {
        "best_checkpoint": best,
        "special_models": special_models,
        "min_trades": min_trades,
        "min_trades_info": {
            "timeframe": timeframe,
            "test_days": test_days_actual if test_days_actual > 0 else int(days * (1 - train_ratio)),
            "test_bars": len(df_test)
        },
        "checkpoint_stats": checkpoint_stats,
        "selection_note": "Best selected from regular checkpoints only. Special models (best/final) shown for comparison."
    }
    with open(selector_config_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved selection result to: {selector_config_path}")
    
    # Возвращаем список всех метрик для последующей визуализации
    return metrics


def _compute_checkpoint_stats(
    regular_checkpoints: list,
    metrics_list: list,
    best_checkpoint: dict,
    model_dir: Path,
    min_trades: int,
) -> dict:
    """Вычисляет агрегатные статистики по всем чекпоинтам для таблицы моделей.
    
    Returns:
        dict с ключами: total_checkpoints, valid_checkpoints, positive_pct,
              plateau_found, plateau_len, tail_profitable_pct, verdict, verdict_reasons
    """
    total = len(regular_checkpoints)
    
    # Valid = прошли порог min_trades
    valid = [m for m in regular_checkpoints if m.get("total_trades", 0) >= min_trades]
    valid_count = len(valid)
    
    # % чекпоинтов с положительным return (среди valid)
    positive_count = sum(1 for m in valid if m.get("total_return_pct", 0) > 0)
    positive_pct = round(positive_count * 100.0 / valid_count, 1) if valid_count > 0 else 0.0
    
    # Tail profitable % (последние 20% valid чекпоинтов)
    valid_sorted = sorted(valid, key=lambda x: x.get("timestep", 0))
    tail_size = max(1, len(valid_sorted) // 5)  # 20%
    tail = valid_sorted[-tail_size:]
    tail_profitable = sum(
        1 for m in tail
        if m.get("total_return_pct", 0) > 0 and m.get("profit_factor", 0) >= 1.0
    )
    tail_profitable_pct = round(tail_profitable * 100.0 / len(tail), 1) if tail else 0.0
    
    # Plateau и verdict через существующие функции
    best_name = best_checkpoint.get("checkpoint", "")
    thresholds = load_thresholds_from_config(model_dir)
    plateau = compute_plateau(metrics_list, best_name, thresholds)
    verdict_result = compute_live_verdict(metrics_list, best_name, thresholds, plateau)
    
    return {
        "total_checkpoints": total,
        "valid_checkpoints": valid_count,
        "positive_pct": positive_pct,
        "plateau_found": plateau.get("found", False),
        "plateau_len": plateau.get("len", 0),
        "tail_profitable_pct": tail_profitable_pct,
        "verdict": verdict_result.get("status", "UNKNOWN"),
        "verdict_reasons": verdict_result.get("reasons", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Select best RL model checkpoint by trading metrics")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory (e.g. rl_system/models/...) ")
    parser.add_argument("--min-trades", type=int, default=None, 
                       help="Minimum number of trades to consider model valid. "
                            "If not specified, calculates dynamically from timeframe and backtest length (recommended).")
    args = parser.parse_args()

    select_best_checkpoint(Path(args.model_dir), min_trades=args.min_trades)


if __name__ == "__main__":
    main()
