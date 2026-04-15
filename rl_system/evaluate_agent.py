"""
Evaluation и Backtesting RL Агента
===================================

Детальный анализ производительности обученной модели.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import MarketTradingEnv, Positions


def parse_args():
    """Парсинг аргументов."""
    parser = argparse.ArgumentParser(description='Evaluate RL Trading Agent')
    
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained model (e.g., rl_system/models/BTCUSDT_30m_20241102_120000)')
    parser.add_argument('--symbol', type=str, default=None, 
                       help='Symbol (auto-detect from config if None)')
    parser.add_argument('--timeframe', type=str, default=None,
                       help='Timeframe (auto-detect from config if None)')
    parser.add_argument('--days', type=int, default=180, help='Days to evaluate')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--save-results', action='store_true', help='Save results to CSV/PNG')
    
    return parser.parse_args()


def load_model_and_config(model_path: str):
    """Загружает модель и конфигурацию."""
    model_dir = Path(model_path)
    
    # Загружаем конфиг
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Загружаем модель (пробуем best, потом final)
    best_model_path = model_dir / "best" / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"
    
    if best_model_path.exists():
        print(f"📦 Loading best model from {best_model_path}")
        model = PPO.load(best_model_path)
    elif final_model_path.exists():
        print(f"📦 Loading final model from {final_model_path}")
        model = PPO.load(final_model_path)
    else:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Загружаем VecNormalize если есть (но не инициализируем с dummy env)
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    vec_normalize_stats = None
    if vec_normalize_path.exists():
        print(f"📦 Loading VecNormalize from {vec_normalize_path}")
        # Сохраняем путь для загрузки позже с реальным окружением
        vec_normalize_stats = vec_normalize_path
    
    return model, config, vec_normalize_stats


def evaluate_agent(model, env, vec_normalize=None, n_episodes=1, deterministic=True, render=False):
    """
    Оценивает агента на окружении.
    
    Returns:
        results: Dict с метриками
        episode_data: List[Dict] с данными по каждому эпизоду
    """
    if vec_normalize:
        env = vec_normalize
        env.training = False
        env.norm_reward = False
    
    episode_data = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        episode_info = {
            'equity_history': [],
            'actions': [],
            'positions': [],
            'prices': [],
            'rewards': [],
            'trades': []
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            # Сохраняем данные
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list):
                info = info[0]
            
            episode_info['equity_history'].append(info.get('equity', 0))
            episode_info['actions'].append(action[0] if isinstance(action, np.ndarray) else action)
            episode_info['positions'].append(info.get('position', 'FLAT'))
            episode_info['rewards'].append(reward[0] if isinstance(reward, np.ndarray) else reward)
            
            step_count += 1
            
            if render and step_count % 50 == 0:
                print(f"\nStep {step_count}")
                print(f"  Equity: ${info.get('equity', 0):.2f}")
                print(f"  Position: {info.get('position', 'FLAT')}")
                print(f"  Trades: {info.get('total_trades', 0)}")
        
        # Финальная статистика эпизода
        episode_data.append({
            'episode': episode,
            'steps': step_count,
            'final_equity': info.get('equity', 0),
            'total_return': info.get('return', 0),
            'total_trades': info.get('total_trades', 0),
            'win_rate': info.get('win_rate', 0),
            'max_drawdown': info.get('max_drawdown', 0),
            'details': episode_info
        })
        
        # Статистика действий
        actions = episode_info['actions']
        action_counts = {
            'SELL': actions.count(0),
            'HOLD': actions.count(1),
            'BUY': actions.count(2)
        }
        total_actions = len(actions)
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes} completed")
        print(f"  Final Equity: ${info.get('equity', 0):.2f}")
        print(f"  Return: {info.get('return', 0) * 100:.2f}%")
        print(f"  Trades: {info.get('total_trades', 0)}")
        print(f"  Win Rate: {info.get('win_rate', 0) * 100:.1f}%")
        print(f"  Max Drawdown: {info.get('max_drawdown', 0) * 100:.2f}%")
        print(f"\n  📊 Action Distribution:")
        print(f"     SELL: {action_counts['SELL']} ({action_counts['SELL']/total_actions*100:.1f}%)")
        print(f"     HOLD: {action_counts['HOLD']} ({action_counts['HOLD']/total_actions*100:.1f}%)")
        print(f"     BUY:  {action_counts['BUY']} ({action_counts['BUY']/total_actions*100:.1f}%)")
        print(f"{'='*60}")
    
    # Агрегированные результаты
    results = {
        'n_episodes': n_episodes,
        'avg_return': np.mean([ep['total_return'] for ep in episode_data]),
        'std_return': np.std([ep['total_return'] for ep in episode_data]),
        'avg_trades': np.mean([ep['total_trades'] for ep in episode_data]),
        'avg_win_rate': np.mean([ep['win_rate'] for ep in episode_data]),
        'avg_max_drawdown': np.mean([ep['max_drawdown'] for ep in episode_data]),
        'min_return': min([ep['total_return'] for ep in episode_data]),
        'max_return': max([ep['total_return'] for ep in episode_data]),
    }
    
    return results, episode_data


def plot_results(episode_data, save_path=None):
    """Визуализация результатов."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for ep_data in episode_data:
        details = ep_data['details']
        
        # 1. Equity curve
        axes[0].plot(details['equity_history'], label=f"Episode {ep_data['episode']}", alpha=0.7)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Equity (USDT)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Actions distribution
        actions = details['actions']
        action_names = ['SELL', 'HOLD', 'BUY']
        action_counts = [actions.count(i) for i in range(3)]
        axes[1].bar(action_names, action_counts, alpha=0.7)
        axes[1].set_title('Action Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative rewards
        cumulative_rewards = np.cumsum(details['rewards'])
        axes[2].plot(cumulative_rewards, label=f"Episode {ep_data['episode']}", alpha=0.7)
        axes[2].set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Cumulative Reward')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()


def save_results_to_csv(episode_data, results, save_dir):
    """Сохраняет результаты в CSV."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary_df = pd.DataFrame([results])
    summary_path = save_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📄 Summary saved to {summary_path}")
    
    # Episode details
    episodes_df = pd.DataFrame([{
        'episode': ep['episode'],
        'steps': ep['steps'],
        'final_equity': ep['final_equity'],
        'return': ep['total_return'],
        'trades': ep['total_trades'],
        'win_rate': ep['win_rate'],
        'max_drawdown': ep['max_drawdown']
    } for ep in episode_data])
    
    episodes_path = save_dir / "episodes.csv"
    episodes_df.to_csv(episodes_path, index=False)
    print(f"📄 Episodes saved to {episodes_path}")


def main():
    """Entry point."""
    args = parse_args()
    
    # Загружаем модель и конфиг
    model, config, vec_normalize = load_model_and_config(args.model_path)
    
    # Используем конфиг если параметры не заданы
    symbol = args.symbol or config.get('symbol', 'BTCUSDT')
    timeframe = args.timeframe or config.get('timeframe', '30m')
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating RL Agent")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Days: {args.days}")
    print(f"Deterministic: {args.deterministic}")
    print(f"{'='*60}\n")
    
    # Загружаем данные
    print(f"📊 Loading data...")
    loader = DataLoader()
    df = loader.load_data(symbol, timeframe, args.days)
    
    # Feature engineering
    print(f"🔧 Calculating features...")
    engineer = FeatureEngineer()
    df_features = engineer.calculate_features(df)
    df_features = engineer.normalize_features(df_features, method='zscore')
    
    # Создаем окружение
    print(f"🎮 Creating environment...")
    env = MarketTradingEnv(
        df=df_features,
        initial_balance=config.get('initial_balance', 10000),
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
        lookback_window=config.get('lookback', 60),
        position_size=config.get('position_size', 0.95),
        enable_short=config.get('enable_short', False)
    )
    
    env = DummyVecEnv([lambda: env])
    
    # Загружаем VecNormalize если есть
    if vec_normalize:
        print(f"🔧 Applying VecNormalize...")
        vec_normalize_obj = VecNormalize.load(vec_normalize, env)
        vec_normalize_obj.training = False
        vec_normalize_obj.norm_reward = False
        env = vec_normalize_obj
    
    # Оцениваем
    print(f"\n🚀 Starting evaluation...\n")
    results, episode_data = evaluate_agent(
        model, 
        env, 
        vec_normalize=None,  # Уже применён выше
        n_episodes=1,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Выводим результаты
    print(f"\n{'='*60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Return: {results['avg_return'] * 100:.2f}%")
    print(f"Std Return: {results['std_return'] * 100:.2f}%")
    print(f"Min/Max Return: {results['min_return'] * 100:.2f}% / {results['max_return'] * 100:.2f}%")
    print(f"Average Trades: {results['avg_trades']:.0f}")
    print(f"Average Win Rate: {results['avg_win_rate'] * 100:.1f}%")
    print(f"Average Max DD: {results['avg_max_drawdown'] * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # Сохраняем результаты
    if args.save_results:
        results_dir = Path(args.model_path) / "evaluation_results"
        
        # CSV
        save_results_to_csv(episode_data, results, results_dir)
        
        # Plot
        plot_path = results_dir / "evaluation_plot.png"
        plot_results(episode_data, save_path=plot_path)
        
        print(f"\n✅ Results saved to {results_dir}")
    else:
        # Просто показываем график
        plot_results(episode_data)


if __name__ == "__main__":
    main()
