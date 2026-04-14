"""
Оценка стабильности RL моделей (Walk-Forward Analysis)
========================================================

Разбивает test период на N равных частей и оценивает модель на каждой.
Вычисляет метрики стабильности:
- Variance returns (насколько прыгают результаты)
- Sharpe Ratio (риск-adjusted return)
- Maximum Drawdown (максимальное падение)
- Win Rate consistency (стабильность винрейта)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3
import json
from typing import Dict, List, Tuple

sys.path.append('rl_system')
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import CryptoTradingEnv


def load_model(model_path: str):
    """Загружает модель по пути."""
    model_path = Path(model_path)
    
    # Определяем алгоритм по config.json
    config_path = model_path.parent / 'config.json'
    if not config_path.exists():
        config_path = model_path.parent / 'best' / 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    algorithm = config.get('algorithm', 'PPO')
    
    # Загружаем модель
    algo_class = {'PPO': PPO, 'A2C': A2C, 'SAC': SAC, 'TD3': TD3}[algorithm]
    model = algo_class.load(str(model_path))
    
    return model, config


def evaluate_on_period(model, df: pd.DataFrame, config: dict, period_name: str) -> Dict:
    """Оценивает модель на конкретном временном периоде."""
    env = CryptoTradingEnv(
        df=df,
        initial_balance=config.get('initial_balance', 10000),
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
        lookback_window=config.get('lookback', 60),
        position_size=config.get('position_size', 0.3),
        enable_short=config.get('enable_short', True),
        stop_loss_pct=config.get('stop_loss_pct', 0.0),
        max_holding_bars=config.get('max_holding_bars', 0)
    )
    
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    # Собираем метрики
    info = env._get_info()
    total_return = (env.equity - env.initial_balance) / env.initial_balance * 100
    
    # Вычисляем Sharpe Ratio (упрощенно)
    # В идеале нужны дневные returns, но для быстрой оценки:
    sharpe = total_return / max(abs(total_return), 1) if env.total_trades > 0 else 0
    
    # Преобразуем индекс в datetime если нужно
    if hasattr(df.index[0], 'strftime'):
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
    else:
        start_date = f"Bar {df.index[0]}"
        end_date = f"Bar {df.index[-1]}"
    
    return {
        'period': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'days': len(df),
        'return_pct': total_return,
        'final_balance': env.equity,
        'trades': env.total_trades,
        'win_rate': env.winning_trades / env.total_trades * 100 if env.total_trades > 0 else 0,
        'profit_factor': env.gross_profit / env.gross_loss if env.gross_loss > 0 else 0,
        'gross_profit': env.gross_profit,
        'gross_loss': env.gross_loss,
        'sharpe_estimate': sharpe,
        'episode_reward': episode_reward
    }


def calculate_stability_metrics(results: List[Dict]) -> Dict:
    """Вычисляет метрики стабильности по результатам на разных периодах."""
    returns = [r['return_pct'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    trades = [r['trades'] for r in results]
    
    # Variance returns (чем меньше, тем стабильнее)
    return_variance = np.var(returns)
    return_std = np.std(returns)
    
    # Mean return
    mean_return = np.mean(returns)
    
    # Sharpe Ratio (упрощенный: mean / std)
    sharpe_ratio = mean_return / return_std if return_std > 0 else 0
    
    # Win Rate consistency (std win rate)
    wr_variance = np.var(win_rates)
    wr_std = np.std(win_rates)
    
    # Проверяем что модель торгует на всех периодах
    periods_with_trades = sum(1 for t in trades if t > 0)
    trade_consistency = periods_with_trades / len(trades) * 100
    
    # Считаем сколько периодов были прибыльными
    profitable_periods = sum(1 for r in returns if r > 0)
    profitability_rate = profitable_periods / len(returns) * 100
    
    # Maximum Drawdown (между периодами)
    cumulative_returns = [10000]  # начальный баланс
    for r in returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r/100))
    
    peak = cumulative_returns[0]
    max_dd = 0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'mean_return': mean_return,
        'return_std': return_std,
        'return_variance': return_variance,
        'sharpe_ratio': sharpe_ratio,
        'mean_win_rate': np.mean(win_rates),
        'win_rate_std': wr_std,
        'trade_consistency': trade_consistency,
        'profitability_rate': profitability_rate,
        'max_drawdown': max_dd,
        'total_trades': sum(trades),
        'periods_evaluated': len(results)
    }


def print_results(model_name: str, period_results: List[Dict], stability_metrics: Dict):
    """Красиво выводит результаты."""
    print(f"\n{'='*80}")
    print(f"📊 WALK-FORWARD ANALYSIS: {model_name}")
    print(f"{'='*80}\n")
    
    print("📈 Results by Period:")
    print(f"{'─'*80}")
    print(f"{'Period':<12} {'Dates':<25} {'Days':<6} {'Return':<10} {'Trades':<8} {'WR':<8} {'PF':<8}")
    print(f"{'─'*80}")
    
    for r in period_results:
        print(f"{r['period']:<12} {r['start_date']} to {r['end_date']:<10} "
              f"{r['days']:<6} {r['return_pct']:>+7.2f}% {r['trades']:>6} "
              f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f}")
    
    print(f"{'─'*80}\n")
    
    print("🎯 Stability Metrics:")
    print(f"{'─'*80}")
    m = stability_metrics
    print(f"   Mean Return:          {m['mean_return']:>+7.2f}% (avg across periods)")
    print(f"   Return Std Dev:       {m['return_std']:>7.2f}% (lower = more stable)")
    print(f"   Sharpe Ratio:         {m['sharpe_ratio']:>7.2f} (higher = better risk-adjusted)")
    print(f"   Max Drawdown:         {m['max_drawdown']:>7.2f}% (lower = safer)")
    print()
    print(f"   Mean Win Rate:        {m['mean_win_rate']:>7.1f}%")
    print(f"   Win Rate Std Dev:     {m['win_rate_std']:>7.1f}% (lower = more consistent)")
    print()
    print(f"   Trade Consistency:    {m['trade_consistency']:>7.1f}% (periods with trades)")
    print(f"   Profitability Rate:   {m['profitability_rate']:>7.1f}% (profitable periods)")
    print(f"   Total Trades:         {m['total_trades']:>7} (all periods)")
    print(f"{'─'*80}\n")
    
    # Оценка стабильности
    print("💡 Stability Assessment:")
    score = 0
    
    if m['return_std'] < 10:
        print("   ✅ Low return variance - STABLE")
        score += 2
    elif m['return_std'] < 20:
        print("   ⚠️  Medium return variance - MODERATE")
        score += 1
    else:
        print("   ❌ High return variance - UNSTABLE")
    
    if m['sharpe_ratio'] > 1.5:
        print("   ✅ Excellent Sharpe Ratio - STRONG")
        score += 2
    elif m['sharpe_ratio'] > 0.8:
        print("   ⚠️  Good Sharpe Ratio - ACCEPTABLE")
        score += 1
    else:
        print("   ❌ Low Sharpe Ratio - WEAK")
    
    if m['profitability_rate'] >= 75:
        print("   ✅ High profitability rate - RELIABLE")
        score += 2
    elif m['profitability_rate'] >= 50:
        print("   ⚠️  Medium profitability rate - ACCEPTABLE")
        score += 1
    else:
        print("   ❌ Low profitability rate - UNRELIABLE")
    
    if m['max_drawdown'] < 10:
        print("   ✅ Low drawdown - SAFE")
        score += 2
    elif m['max_drawdown'] < 20:
        print("   ⚠️  Medium drawdown - MODERATE RISK")
        score += 1
    else:
        print("   ❌ High drawdown - RISKY")
    
    print(f"\n   Overall Stability Score: {score}/8")
    if score >= 7:
        print("   🏆 EXCELLENT - Production ready!")
    elif score >= 5:
        print("   ✅ GOOD - Acceptable for production")
    elif score >= 3:
        print("   ⚠️  FAIR - Use with caution")
    else:
        print("   ❌ POOR - Not recommended for production")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL model stability using Walk-Forward Analysis')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to model (e.g., rl_system/models/BTCUSDT_1d_PPO_730d_.../best/best_model.zip)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1d', 
                       choices=['15m', '30m', '1h', '4h', '1d'],
                       help='Timeframe')
    parser.add_argument('--days', type=int, default=730, help='Total days of data')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split')
    parser.add_argument('--n-periods', type=int, default=2, 
                       help='Number of periods to split test into (default: 2, will be auto-adjusted if test too small)')
    
    args = parser.parse_args()
    
    print(f"\n🔍 Loading model from: {args.model_path}")
    model, config = load_model(args.model_path)
    model_name = Path(args.model_path).parent.parent.name
    
    print(f"📊 Loading data: {args.symbol} {args.timeframe} ({args.days} days)")
    
    # Загружаем данные
    loader = DataLoader()
    df = loader.load_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days
    )
    
    # Feature engineering
    engineer = FeatureEngineer()
    df = engineer.calculate_features(df)
    
    # Разбиваем на train/test
    split_idx = int(len(df) * args.train_ratio)
    test_df = df[split_idx:].copy()
    
    print(f"   Train: {split_idx} bars")
    print(f"   Test: {len(test_df)} bars")
    print(f"   Splitting test into {args.n_periods} periods...\n")
    
    # Разбиваем test на N периодов
    period_size = len(test_df) // args.n_periods
    period_results = []
    
    # Проверяем что периоды достаточно большие (минимум lookback * 1.5)
    min_period_size = int(config.get('lookback', 60) * 1.5)
    if period_size < min_period_size:
        print(f"⚠️  Warning: Period size ({period_size}) too small for lookback ({config.get('lookback', 60)})")
        print(f"   Reducing number of periods to ensure minimum {min_period_size} bars per period")
        args.n_periods = max(1, len(test_df) // min_period_size)
        period_size = len(test_df) // args.n_periods
        print(f"   New number of periods: {args.n_periods}\n")
    
    for i in range(args.n_periods):
        start_idx = i * period_size
        # Последний период берет все оставшиеся данные
        end_idx = (i + 1) * period_size if i < args.n_periods - 1 else len(test_df)
        
        period_df = test_df.iloc[start_idx:end_idx].copy()
        period_name = f"Period {i+1}"
        
        print(f"⏳ Evaluating {period_name} ({len(period_df)} bars)...")
        result = evaluate_on_period(model, period_df, config, period_name)
        period_results.append(result)
    
    # Вычисляем метрики стабильности
    stability_metrics = calculate_stability_metrics(period_results)
    
    # Выводим результаты
    print_results(model_name, period_results, stability_metrics)
    
    # Сохраняем в JSON
    output_dir = Path(args.model_path).parent.parent / 'stability_analysis'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'walk_forward_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_name': model_name,
            'model_path': str(args.model_path),
            'config': config,
            'test_params': {
                'symbol': args.symbol,
                'timeframe': args.timeframe,
                'days': args.days,
                'train_ratio': args.train_ratio,
                'n_periods': args.n_periods
            },
            'period_results': period_results,
            'stability_metrics': stability_metrics
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
