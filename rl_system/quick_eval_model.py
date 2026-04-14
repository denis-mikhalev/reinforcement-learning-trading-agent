"""
Быстрая оценка RL модели
========================
Проверяет РЕАЛЬНОЕ качество модели на test данных.
"""

import json
from pathlib import Path
import sys

sys.path.append('rl_system')
from stable_baselines3 import PPO, A2C, SAC, TD3

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import CryptoTradingEnv


def evaluate_model(model_path: str):
    """Оценивает модель на test данных."""
    model_dir = Path(model_path)
    
    # Загружаем конфиг
    config_path = model_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"📊 Evaluating Model: {model_dir.name}")
    print(f"{'='*70}")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Symbol: {config['symbol']}")
    print(f"Timeframe: {config['timeframe']}")
    print(f"Days: {config['days']}")
    print(f"Timesteps: {config['total_timesteps']:,}")
    print(f"{'='*70}\n")
    
    # Загружаем модель (пробуем best, потом final)
    best_model_path = model_dir / "best" / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"
    
    algorithm = config['algorithm']
    algorithm_classes = {
        'PPO': PPO,
        'A2C': A2C,
        'SAC': SAC,
        'TD3': TD3
    }
    
    model_class = algorithm_classes.get(algorithm)
    if not model_class:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    if best_model_path.exists():
        print(f"📦 Loading BEST model...")
        model = model_class.load(best_model_path)
        model_type = "best"
    elif final_model_path.exists():
        print(f"📦 Loading FINAL model...")
        model = model_class.load(final_model_path)
        model_type = "final"
    else:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Загружаем данные (те же, что использовались при обучении)
    print("📊 Loading data...")
    loader = DataLoader(cache_dir="data/cache")
    df = loader.load_data(
        symbol=config['symbol'],
        timeframe=config['timeframe'],
        days=config['days']
    )
    
    # Рассчитываем features ТОЧНО так же, как в train_agent_v2.py
    print("🔧 Calculating features (enhanced with volume profile & order flow)...")
    engineer = FeatureEngineer()
    df = engineer.calculate_features(df)
    
    # Опционально добавляем Multi-Timeframe features, если они использовались при обучении
    use_mtf = config.get('hyperparameters', {}).get('use_mtf', None)
    # В тренере флаг хранится как отдельный аргумент, поэтому подстрахуемся и посмотрим исходный конфиг
    if use_mtf is None:
        # try top-level key as in train config / CLI
        use_mtf = config.get('use_mtf', False)
    
    if use_mtf:
        print("🔧 Adding Multi-Timeframe features (Weekly/Monthly)...")
        df = engineer.add_multi_timeframe_features(df)
        print(f"✅ Total features: {len(engineer.get_feature_names(df))}")
    else:
        print("ℹ️  Multi-Timeframe features disabled for evaluation")
    
    # Нормализация (как при обучении)
    print("📊 Normalizing features...")
    df = engineer.normalize_features(df, method='zscore')
    
    # Разделяем на train/test (80/20)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"✅ Test data: {len(test_df)} bars\n")
    
    # Создаем test окружение
    print(f"🎮 Creating test environment...")
    test_env = CryptoTradingEnv(
        df=test_df,
        initial_balance=10000.0,
        commission=config['commission'],
        slippage=config['slippage'],
        lookback_window=config['lookback'],
        position_size=1.0,  # 100% для чистой оценки
        enable_short=config.get('enable_short', True)
    )
    
    # Запускаем evaluation
    print(f"🚀 Running evaluation ({model_type} model)...\n")
    obs, _ = test_env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    # Получаем финальные метрики
    final_info = test_env._get_info()
    
    # Выводим результаты
    print(f"{'='*70}")
    print(f"📊 Evaluation Results ({model_type.upper()} model)")
    print(f"{'='*70}")
    print(f"Return:          {final_info['total_return_pct']:+.2f}%")
    print(f"Final Balance:   ${test_env.balance:,.2f}")
    print(f"Final Equity:    ${test_env.equity:,.2f}")
    print(f"Total Trades:    {final_info['total_trades']}")
    print(f"Winning Trades:  {final_info['winning_trades']}")
    print(f"Losing Trades:   {final_info['losing_trades']}")
    print(f"Win Rate:        {final_info['win_rate']*100:.1f}%")
    print(f"Profit Factor:   {final_info['profit_factor']:.2f}")
    print(f"Gross Profit:    ${final_info['gross_profit']:,.2f}")
    print(f"Gross Loss:      ${final_info['gross_loss']:,.2f}")
    print(f"Max Drawdown:    {final_info['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:    {final_info['sharpe_ratio']:.2f}")
    print(f"Episode Reward:  {episode_reward:.2f}")
    print(f"{'='*70}\n")
    
    return {
        'model_type': model_type,
        'return_pct': final_info['total_return_pct'],
        'total_trades': final_info['total_trades'],
        'win_rate': final_info['win_rate'],
        'profit_factor': final_info['profit_factor'],
        'sharpe_ratio': final_info['sharpe_ratio'],
        'episode_reward': episode_reward
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_eval_model.py <model_path>")
        print("Example: python quick_eval_model.py rl_system/models/BTCUSDT_4h_A2C_20251104_113800")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results = evaluate_model(model_path)
