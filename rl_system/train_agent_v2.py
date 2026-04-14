"""
Улучшенное Обучение RL Агента (v2)
====================================

Поддержка:
- Нескольких алгоритмов: PPO, SAC, TD3, A2C
- Расширенных features (volume profile, order flow)
- Больших таймфреймов (1h, 4h)
- Гиперпараметров под каждый алгоритм
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Stable-Baselines3 algorithms
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import csv

import sys
# Fix encoding for Windows console to support emoji/unicode
if sys.platform == 'win32':
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8')

sys.path.append('rl_system')
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import CryptoTradingEnv
from select_best_model import select_best_checkpoint
from plateau_analysis import compute_plateau, compute_live_verdict, load_thresholds_from_config
from model_quality_assessment import assess_model_quality, format_quality_assessment_markdown

# Импорт для отправки уведомлений в Telegram
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root to path
    from telegram_sender import send_telegram_message
    TELEGRAM_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Telegram notifications unavailable: {e}")
    send_telegram_message = None
    TELEGRAM_AVAILABLE = False


def format_training_completion_message(config, model_dir, final_info, best_info, best_checkpoint_name, duration, args):
    """
    Форматирует сообщение о завершении обучения для Telegram
    
    Args:
        config: конфигурация модели
        model_dir: путь к директории модели
        final_info: метрики финальной модели
        best_info: метрики лучшей модели
        best_checkpoint_name: имя лучшего чекпоинта
        duration: длительность обучения в секундах
        args: аргументы командной строки
    
    Returns:
        str: отформатированное сообщение для Telegram
    """
    # Определяем эмодзи для статуса
    status_emoji = "✅"
    if best_info and best_info['total_return_pct'] < 0:
        status_emoji = "⚠️"
    elif not best_info:
        status_emoji = "❌"
    
    # Форматируем длительность
    duration_minutes = duration / 60
    duration_hours = duration / 3600
    if duration_hours >= 1:
        duration_str = f"{duration_hours:.1f} hours ({duration_minutes:.0f} min)"
    else:
        duration_str = f"{duration_minutes:.1f} minutes"
    
    # Базовая информация
    message = f"""{status_emoji} <b>TRAINING COMPLETED</b>

"""
    
    message += f"""🤖 <b>Model Information:</b>
<b>Symbol:</b> {args.symbol}
<b>Timeframe:</b> {args.timeframe}
<b>Algorithm:</b> {args.algorithm}
<b>Model Name:</b> {args.model_name}
"""
    
    message += f"""\n⏱️ <b>Training Duration:</b> {duration_str}
"""
    
    # Информация о лучшем чекпоинте
    if best_info:
        message += f"""\n📊 <b>Best Checkpoint (by trading metrics):</b>
<b>Checkpoint:</b> {best_checkpoint_name or 'N/A'}
<b>Return:</b> {best_info['total_return_pct']:+.2f}%
<b>Win Rate:</b> {best_info['win_rate']*100:.1f}%
<b>Trades:</b> {best_info['total_trades']}
<b>Profit Factor:</b> {best_info['profit_factor']:.2f}
<b>Max Drawdown:</b> {best_info['max_drawdown_pct']:.2f}%
<b>Sharpe Ratio:</b> {best_info['sharpe_ratio']:.2f}
"""
    
    # Информация о финальной модели для сравнения
    if final_info:
        message += f"""\n📈 <b>Final Model (last step):</b>
<b>Return:</b> {final_info['total_return_pct']:+.2f}%
<b>Win Rate:</b> {final_info['win_rate']*100:.1f}%
<b>Trades:</b> {final_info['total_trades']}
<b>Profit Factor:</b> {final_info['profit_factor']:.2f}
"""
    
    # Анализ переобучения
    if best_info and final_info:
        return_diff = abs(final_info['total_return_pct'] - best_info['total_return_pct'])
        if return_diff < 3:
            overfitting_status = "✅ Minimal overfitting"
        elif return_diff < 10:
            overfitting_status = "⚠️ Moderate overfitting"
        else:
            overfitting_status = "❌ Strong overfitting"
        
        message += f"""\n🔍 <b>Overfitting Analysis:</b>
<b>Difference:</b> {return_diff:.2f}%
<b>Status:</b> {overfitting_status}
"""
    
    # Путь к модели
    message += f"""\n💾 <b>Model Path:</b>
<code>{model_dir.name}</code>
"""
    
    message += "\n🎉 <b>Agent is ready for live trading!</b>"
    
    return message


class TrainingLogCallback(BaseCallback):
    """Callback для записи метрик обучения в Markdown файл"""
    def __init__(self, log_file: Path, n_steps: int = 32, log_interval_multiplier: int = 1, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.last_log_step = 0
        # Логируем с той же частотой что и консоль (n_steps) или реже (через multiplier)
        # Например: n_steps=32, multiplier=1 -> каждые 32 шага (как консоль)
        #           n_steps=32, multiplier=10 -> каждые 320 шагов
        self.log_interval = n_steps * log_interval_multiplier
        print(f"📝 Training log interval: every {self.log_interval} timesteps (n_steps={n_steps} × {log_interval_multiplier})")
        
    def _on_step(self) -> bool:
        # Записываем метрики точно на тех же шагах, что и консоль
        # Проверяем: текущий timestep кратен log_interval И мы еще не записывали этот шаг
        if self.num_timesteps % self.log_interval == 0 and self.num_timesteps != self.last_log_step:
            self.last_log_step = self.num_timesteps
            
            # Получаем метрики из logger
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                metrics = self.model.logger.name_to_value
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"### Timestep {self.num_timesteps:,}\n\n")
                    f.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Rollout metrics
                    if any(k.startswith('rollout/') for k in metrics.keys()):
                        f.write(f"**Rollout Metrics:**\n\n")
                        for key, value in sorted(metrics.items()):
                            if key.startswith('rollout/'):
                                metric_name = key.replace('rollout/', '')
                                f.write(f"- {metric_name}: {value:.2f}\n")
                        f.write("\n")
                    
                    # Eval metrics (появляются во время evaluation)
                    if any(k.startswith('eval/') for k in metrics.keys()):
                        f.write(f"**Eval Metrics:**\n\n")
                        for key, value in sorted(metrics.items()):
                            if key.startswith('eval/'):
                                metric_name = key.replace('eval/', '')
                                f.write(f"- {metric_name}: {value:.2f}\n")
                        f.write("\n")
                    
                    # Time metrics
                    if any(k.startswith('time/') for k in metrics.keys()):
                        f.write(f"**Time Metrics:**\n\n")
                        for key, value in sorted(metrics.items()):
                            if key.startswith('time/'):
                                metric_name = key.replace('time/', '')
                                if metric_name == 'time_elapsed':
                                    f.write(f"- {metric_name}: {value/60:.1f} minutes\n")
                                else:
                                    f.write(f"- {metric_name}: {value:.2f}\n")
                        f.write("\n")
                    
                    # Training metrics
                    if any(k.startswith('train/') for k in metrics.keys()):
                        f.write(f"**Training Metrics:**\n\n")
                        for key, value in sorted(metrics.items()):
                            if key.startswith('train/'):
                                metric_name = key.replace('train/', '')
                                f.write(f"- {metric_name}: {value:.4f}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
        
        return True


class DetailedEvalCallback(EvalCallback):
    """
    Расширенный callback для вывода детальных метрик во время eval
    Показывает: balance, profit, trades, win_rate во время обучения
    Сохраняет историю eval для анализа траектории
    """
    def __init__(self, *args, log_file: Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_history = []  # История всех eval: {timestep, return, trades, ...}
        self.log_file = log_file  # Добавляем поддержку логирования в файл
    
    def _on_step(self) -> bool:
        # Вызываем родительский метод (сохраняет best model и т.д.)
        continue_training = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Записываем базовые eval метрики из parent callback в файл
            if self.log_file and len(self.evaluations_results) > 0:
                # evaluations_results[-1] это список наград, берем среднее
                last_mean_reward = np.mean(self.evaluations_results[-1])
                last_ep_length = np.mean(self.evaluations_length[-1]) if len(self.evaluations_length) > 0 else 0
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"### 📊 Base Evaluation at Timestep {self.num_timesteps:,}\n\n")
                    f.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"**Base Metrics:**\n\n")
                    f.write(f"- **mean_reward:** {last_mean_reward:.2f}\n")
                    f.write(f"- **mean_ep_length:** {last_ep_length:.2f}\n")
                    f.write(f"\nEval num_timesteps={self.num_timesteps}, episode_reward={last_mean_reward:.2f} +/- 0.00\n")
                    f.write(f"Episode length: {last_ep_length:.2f} +/- 0.00\n\n")
            
            # Получаем метрики из eval environment
            # eval_env это DummyVecEnv, достаем реальный env
            real_env = self.eval_env.envs[0]
            
            # Проверяем что env это наш CryptoTradingEnv (не Monitor wrapper)
            if hasattr(real_env, 'env') and isinstance(real_env.env, CryptoTradingEnv):
                trading_env = real_env.env
            elif isinstance(real_env, CryptoTradingEnv):
                trading_env = real_env
            else:
                # Не можем получить метрики
                return continue_training
            
            # ВАЖНО: EvalCallback уже сбросил env после своей оценки,
            # поэтому total_trades = 0. Запускаем свой eval эпизод для метрик.
            obs, _ = trading_env.reset()
            done = False
            while not done:
                # Используем VecEnv для predict (требует batch dimension)
                obs_batch = obs.reshape(1, -1) if len(obs.shape) == 1 else obs[np.newaxis, :]
                action, _ = self.model.predict(obs_batch, deterministic=True)
                action = action[0] if isinstance(action, np.ndarray) else action
                obs, reward, terminated, truncated, info = trading_env.step(action)
                done = terminated or truncated
            
            # Теперь получаем метрики после полного эпизода
            total_trades = trading_env.total_trades
            if total_trades > 0:
                win_rate = trading_env.winning_trades / total_trades
                profit_factor = trading_env.gross_profit / trading_env.gross_loss if trading_env.gross_loss > 0 else 0.0
                total_return_pct = (trading_env.equity - trading_env.initial_balance) / trading_env.initial_balance * 100
                max_drawdown_pct = trading_env.max_drawdown * 100
                avg_profit_per_trade = (trading_env.equity - trading_env.initial_balance) / total_trades
                
                # Сохраняем в историю
                self.eval_history.append({
                    'timestep': self.num_timesteps,
                    'return_pct': round(total_return_pct, 2),
                    'trades': total_trades,
                    'win_rate': round(win_rate * 100, 2),
                    'profit_factor': round(profit_factor, 2),
                    'balance': round(trading_env.balance, 2),
                    'max_drawdown_pct': round(max_drawdown_pct, 2),
                    'reward_total': round(float(getattr(trading_env, 'reward_total_sum', 0.0)), 4),
                    'reward_steps': int(getattr(trading_env, 'reward_steps', 0)),
                    'reward_breakdown': getattr(trading_env, 'reward_breakdown_sums', {}),
                })
                
                # Записываем в Markdown лог, если файл указан
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"### 📊 Evaluation at Timestep {self.num_timesteps:,}\n\n")
                        f.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write(f"**Trading Performance:**\n\n")
                        f.write(f"- **Return:** {total_return_pct:+.2f}%\n")
                        f.write(f"- **Trades:** {total_trades}\n")
                        f.write(f"- **Win Rate:** {win_rate*100:.1f}%\n")
                        f.write(f"- **Profit Factor:** {profit_factor:.2f}\n")
                        f.write(f"- **Max Drawdown:** {max_drawdown_pct:.2f}%\n")
                        f.write(f"- **Balance:** ${trading_env.balance:,.2f}\n")
                        f.write(f"- **Gross Profit:** ${trading_env.gross_profit:,.2f}\n")
                        f.write(f"- **Gross Loss:** -${trading_env.gross_loss:,.2f}\n")
                        f.write(f"- **Avg Profit/Trade:** ${avg_profit_per_trade:,.2f}\n")

                        # Reward breakdown (суммы за эпизод)
                        rb = getattr(trading_env, 'reward_breakdown_sums', None)
                        if isinstance(rb, dict) and rb:
                            f.write("\n**Reward Breakdown (episode sums):**\n\n")
                            f.write(f"- **reward_total:** {float(getattr(trading_env, 'reward_total_sum', 0.0)):.4f}\n")
                            f.write(f"- **reward_steps:** {int(getattr(trading_env, 'reward_steps', 0))}\n")
                            for key in [
                                'realized_close',
                                'open_penalty',
                                'hold_unrealized',
                                'hold_flat_penalty',
                                'too_few_trades_penalty',
                                'forced_stop_loss_close',
                                'forced_max_holding_close',
                                'forced_end_close',
                                'terminal_bonus',
                                'dd_bonus',
                            ]:
                                if key in rb:
                                    f.write(f"- **{key}:** {float(rb.get(key, 0.0)):.4f}\n")
                        f.write("\n---\n\n")
                
                print(f"📊 Eval Metrics (timestep {self.num_timesteps}):")
                print(f"   💰 Balance: ${trading_env.balance:,.2f} (Return: {total_return_pct:+.2f}%)")
                print(f"   🔄 Trades: {total_trades} | Win Rate: {win_rate*100:.1f}% | PF: {profit_factor:.2f}")
                print(f"   📉 Max Drawdown: {max_drawdown_pct:.2f}%")
                print()
            else:
                # Даже без trades сохраняем точку
                self.eval_history.append({
                    'timestep': self.num_timesteps,
                    'return_pct': 0.0,
                    'trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'balance': trading_env.initial_balance,
                })
                
                # Записываем в лог
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"**Trading Performance:** No trades yet\n\n")
                        f.write("---\n\n")
                
                print(f"📊 Eval Metrics (timestep {self.num_timesteps}): No trades yet")
                print()
        
        return continue_training


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Train RL Trading Agent v2 (Multi-Algorithm)')
    
    # Data arguments
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', 
                       choices=['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'], 
                       help='Timeframe (larger = less noise)')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    # Разбиение можно задать либо через train_ratio, либо через желаемую длину теста в днях
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio (0.8 = 80% train, 20% test)')
    parser.add_argument('--test-days-target', type=int, default=None, help='Desired test period length in days (overrides train_ratio if set)')
    parser.add_argument('--no-cache', action='store_true', help='Force reload data from Binance (ignore cache)')
    parser.add_argument('--max-cache-age', type=int, default=24, help='Max cache age in hours (default: 24h)')
    
    # Environment arguments
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance in USDT')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window size')
    parser.add_argument('--position-size', type=float, default=0.3, help='Position size (0.3 = 30% of capital)')
    parser.add_argument('--enable-short', action='store_true', help='Enable short positions')
    parser.add_argument('--stop-loss', type=float, default=0.0, help='Stop-loss in % of balance (0.01 = 1%, 0 = disabled)')
    parser.add_argument('--max-holding-bars', type=int, default=0, help='Max bars to hold position (0 = disabled). Ex: 48 bars on 4h = 8 days')

    # Reward shaping (optional; defaults preserve current behavior)
    parser.add_argument('--hold-flat-penalty', type=float, default=0.00005,
                       help='Per-step penalty when HOLD while FLAT (default 0.00005 => reward -0.00005)')
    parser.add_argument('--min-trades-target', type=int, default=0,
                       help='If >0, target minimum trades per episode (used only with too-few-trades-penalty)')
    parser.add_argument('--too-few-trades-penalty', type=float, default=0.0,
                       help='If >0, terminal penalty magnitude when trades < min-trades-target (scaled by deficit)')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3', 'A2C'],
                       help='RL algorithm to use')
    
    # Training arguments (common)
    parser.add_argument('--total-timesteps', type=int, default=500000, 
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=None, 
                       help='Learning rate (auto if None)')
    
    # PPO-specific
    parser.add_argument('--n-steps', type=int, default=2048, help='PPO: Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='PPO: Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='PPO: Number of epochs')
    
    # SAC/TD3-specific
    parser.add_argument('--buffer-size', type=int, default=100000, help='SAC/TD3: Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=10000, help='SAC/TD3: Steps before learning')
    parser.add_argument('--tau', type=float, default=0.005, help='SAC/TD3: Target network update rate')
    
    # Common hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--ent-coef', type=float, default=None, 
                       help='Entropy coefficient for exploration (auto if None)')
    parser.add_argument('--clip-range', type=float, default=None,
                       help='PPO: Clipping parameter (auto if None, default 0.2)')
    
    # Architecture arguments
    parser.add_argument('--use-attention', action='store_true', 
                       help='Use Attention-based feature extractor (pure attention)')
    parser.add_argument('--use-cnn-attention', action='store_true',
                       help='Use CNN+Attention hybrid feature extractor')
    parser.add_argument('--attention-heads', type=int, default=4,
                       help='Number of attention heads (must divide feature dim)')
    parser.add_argument('--attention-layers', type=int, default=2,
                       help='Number of attention layers')
    
    # Feature engineering arguments
    parser.add_argument('--use-mtf', action='store_true',
                       help='Add Multi-Timeframe features (Weekly/Monthly indicators)')
    
    # Logging arguments
    parser.add_argument('--model-name', type=str, default=None, help='Model name (auto-generated if None)')
    parser.add_argument('--save-freq', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--eval-freq', type=int, default=50000, help='Evaluate every N steps (auto: after each episode if default)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping on no model improvement')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (evaluations without improvement)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (None = random)')
    
    # Config file support
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file (overrides CLI args)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Get parser defaults to check if arg was explicitly set
        parser_defaults = {
            'days': 730,
            'train_ratio': 0.8,
            'test_days_target': None,
            'no_cache': False,
            'max_cache_age': 24,
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'lookback': 60,
            'position_size': 0.3,
            'stop_loss': 0.0,
            'max_holding_bars': 0,
            'hold_flat_penalty': 0.00005,
            'min_trades_target': 0,
            'too_few_trades_penalty': 0.0,
            'total_timesteps': 500000,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'tau': 0.005,
            'gamma': 0.99,
            'clip_range': None,
            'attention_heads': 4,
            'attention_layers': 2,
            'save_freq': 10000,
            'eval_freq': 50000,
            'verbose': 1,
            'patience': 10
        }
        
        # Override args with config values
        for key, value in config.items():
            # Skip non-parameter fields like 'notes', 'description', etc
            if key in ['model_version', 'description', 'date_created', 'result', 'notes']:
                continue
            
            # Convert snake_case to kebab-case if needed
            arg_key = key.replace('_', '_')
            
            if hasattr(args, arg_key):
                current_value = getattr(args, arg_key)
                
                # Override if:
                # 1. Current value is None (not set)
                # 2. Current value equals default (not explicitly set by user)
                # 3. It's a boolean flag
                if current_value is None or \
                   (arg_key in parser_defaults and current_value == parser_defaults[arg_key]) or \
                   isinstance(value, bool) or \
                   arg_key in ['symbol', 'timeframe', 'algorithm']:
                    setattr(args, arg_key, value)
        
        print(f"📋 Loaded config from: {args.config}")
    
    return args


def get_algorithm_defaults(algorithm: str) -> dict:
    """Возвращает оптимальные гиперпараметры для каждого алгоритма."""
    defaults = {
        'PPO': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,  # Increased for exploration
        },
        'A2C': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
        },
        'SAC': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto',  # Automatic entropy tuning
        },
        'TD3': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        }
    }
    return defaults.get(algorithm, {})


def create_env(df, args, is_eval=False):
    """Создает окружение с Monitor wrapper."""
    env = CryptoTradingEnv(
        df=df,
        initial_balance=args.initial_balance,
        commission=args.commission,
        slippage=args.slippage,
        lookback_window=args.lookback,
        position_size=args.position_size,
        enable_short=args.enable_short,
        stop_loss_pct=args.stop_loss,
        max_holding_bars=args.max_holding_bars,
        hold_flat_penalty=args.hold_flat_penalty,
        min_trades_target=args.min_trades_target,
        too_few_trades_penalty=args.too_few_trades_penalty
    )
    
    # Оборачиваем в Monitor для логирования
    monitor_dir = f"rl_system/logs/{args.model_name}/{'eval' if is_eval else 'train'}_monitor"
    os.makedirs(monitor_dir, exist_ok=True)
    env = Monitor(env, monitor_dir)
    
    return env


def create_model(algorithm: str, env, args, defaults: dict):
    """
    Создает модель выбранного алгоритма с оптимальными параметрами.
    """
    # Используем defaults, но позволяем override через args
    learning_rate = args.learning_rate if args.learning_rate is not None else defaults.get('learning_rate', 3e-4)
    ent_coef = args.ent_coef if args.ent_coef is not None else defaults.get('ent_coef', 0.01)
    
    tensorboard_log = f"rl_system/logs/{args.model_name}/tensorboard"
    
    # Подготовка policy kwargs для Attention extractor (если включен)
    policy_kwargs = {}
    if args.use_attention or args.use_cnn_attention:
        from attention_extractor import AttentionFeatureExtractor, CNNAttentionFeatureExtractor
        
        if args.use_attention:
            print(f"🔍 Using AttentionFeatureExtractor (heads={args.attention_heads}, layers={args.attention_layers})")
            policy_kwargs = {
                'features_extractor_class': AttentionFeatureExtractor,
                'features_extractor_kwargs': {
                    'features_dim': 256,
                    'n_attention_heads': args.attention_heads,
                    'n_attention_layers': args.attention_layers,
                    'dropout': 0.1
                }
            }
        elif args.use_cnn_attention:
            print(f"🔍 Using CNNAttentionFeatureExtractor (heads={args.attention_heads})")
            policy_kwargs = {
                'features_extractor_class': CNNAttentionFeatureExtractor,
                'features_extractor_kwargs': {
                    'features_dim': 256,
                    'n_attention_heads': args.attention_heads,
                    'dropout': 0.1
                }
            }
    
    if algorithm == 'PPO':
        model = PPO(
            policy='MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=defaults.get('gae_lambda', 0.95),
            clip_range=defaults.get('clip_range', 0.2),
            ent_coef=ent_coef,
            verbose=args.verbose,
            tensorboard_log=tensorboard_log,
            device='auto'
        )
    
    elif algorithm == 'A2C':
        model = A2C(
            policy='MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=defaults.get('gae_lambda', 1.0),
            ent_coef=ent_coef,
            vf_coef=defaults.get('vf_coef', 0.25),
            verbose=args.verbose,
            tensorboard_log=tensorboard_log,
            device='auto'
        )
    
    elif algorithm == 'SAC':
        # SAC requires continuous action space - we'll need to modify env later
        # For now, using default SAC params
        model = SAC(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=defaults.get('batch_size', 256),
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=defaults.get('ent_coef', 'auto'),
            verbose=args.verbose,
            tensorboard_log=tensorboard_log,
            device='auto'
        )
    
    elif algorithm == 'TD3':
        # TD3 also requires continuous action space
        # Add action noise for exploration
        n_actions = env.action_space.n if hasattr(env.action_space, 'n') else 1
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = TD3(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=defaults.get('batch_size', 100),
            tau=args.tau,
            gamma=args.gamma,
            action_noise=action_noise,
            policy_delay=defaults.get('policy_delay', 2),
            target_policy_noise=defaults.get('target_policy_noise', 0.2),
            target_noise_clip=defaults.get('target_noise_clip', 0.5),
            verbose=args.verbose,
            tensorboard_log=tensorboard_log,
            device='auto'
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def plot_checkpoints_comparison(metrics_list: list, model_dir: Path, best_checkpoint_name: str):
    """
    Создает график сравнения всех чекпоинтов
    
    Args:
        metrics_list: Список метрик всех чекпоинтов из select_best_checkpoint
        model_dir: Путь к директории модели
        best_checkpoint_name: Имя выбранного лучшего чекпоинта
    """
    if not metrics_list:
        print("⚠️  No checkpoint metrics available for plotting")
        return
    
    # Сортируем по timestep
    metrics_list = sorted(metrics_list, key=lambda x: x.get('timestep', 0))
    
    # Фильтруем только реальные чекпоинты (исключаем best/final с специальными timestep)
    real_metrics = [m for m in metrics_list if m.get('timestep', 0) < 900000000]
    
    if not real_metrics:
        print("⚠️  No real checkpoint metrics available (only best/final models)")
        return
    
    # Извлекаем данные только из реальных чекпоинтов
    timesteps = [m.get('timestep', 0) for m in real_metrics]
    returns = [m.get('total_return_pct', 0) for m in real_metrics]
    profit_factors = [m.get('profit_factor', 0) for m in real_metrics]
    trades = [m.get('total_trades', 0) for m in real_metrics]
    win_rates = [m.get('win_rate_pct', 0) for m in real_metrics]
    scores = [m.get('score', 0) for m in real_metrics]
    checkpoints = [m.get('checkpoint', '') for m in real_metrics]
    
    # Найдем индекс лучшего чекпоинта среди реальных чекпоинтов
    best_idx = None
    for i, cp in enumerate(checkpoints):
        if best_checkpoint_name in cp:
            best_idx = i
            break
    
    # Если best_checkpoint это best_model.zip, ищем его метрики в полном списке
    if best_idx is None:
        best_checkpoint_metrics = next((m for m in metrics_list if best_checkpoint_name in m.get('checkpoint', '')), None)
        if best_checkpoint_metrics:
            # Находим ближайший чекпоинт по метрикам
            best_return = best_checkpoint_metrics.get('total_return_pct', 0)
            best_pf = best_checkpoint_metrics.get('profit_factor', 0)
            
            # Ищем наиболее похожий чекпоинт
            min_diff = float('inf')
            for i, m in enumerate(real_metrics):
                diff = abs(m.get('total_return_pct', 0) - best_return) + abs(m.get('profit_factor', 0) - best_pf)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Checkpoint Quality Comparison', fontsize=16, fontweight='bold')
    
    # 1. Return %
    ax1 = axes[0, 0]
    ax1.plot(timesteps, returns, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    if best_idx is not None:
        ax1.plot(timesteps[best_idx], returns[best_idx], 'r*', markersize=20, 
                label=f'Best: {returns[best_idx]:.2f}%', zorder=5)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Return %')
    ax1.set_title('Return % vs Training Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Profit Factor
    ax2 = axes[0, 1]
    ax2.plot(timesteps, profit_factors, 'o-', linewidth=2, markersize=6, color='#A23B72')
    if best_idx is not None:
        ax2.plot(timesteps[best_idx], profit_factors[best_idx], 'r*', markersize=20,
                label=f'Best: {profit_factors[best_idx]:.2f}', zorder=5)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Profit Factor')
    ax2.set_title('Profit Factor vs Training Steps')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Number of Trades
    ax3 = axes[1, 0]
    ax3.plot(timesteps, trades, 'o-', linewidth=2, markersize=6, color='#F18F01')
    if best_idx is not None:
        ax3.plot(timesteps[best_idx], trades[best_idx], 'r*', markersize=20,
                label=f'Best: {trades[best_idx]}', zorder=5)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Number of Trades')
    ax3.set_title('Trading Activity vs Training Steps')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Win Rate %
    ax4 = axes[1, 1]
    ax4.plot(timesteps, win_rates, 'o-', linewidth=2, markersize=6, color='#6A994E')
    if best_idx is not None:
        ax4.plot(timesteps[best_idx], win_rates[best_idx], 'r*', markersize=20,
                label=f'Best: {win_rates[best_idx]:.1f}%', zorder=5)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Win Rate %')
    ax4.set_title('Win Rate % vs Training Steps')
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Score (Combined Metric)
    ax5 = axes[2, 0]
    ax5.plot(timesteps, scores, 'o-', linewidth=2, markersize=6, color='#BC4B51')
    if best_idx is not None:
        ax5.plot(timesteps[best_idx], scores[best_idx], 'r*', markersize=20,
                label=f'Best: {scores[best_idx]:.2f}', zorder=5)
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Combined Score')
    ax5.set_title('Combined Score vs Training Steps (Selection Criterion)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Summary Table
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Подготовка данных для таблицы
    if best_idx is not None:
        summary_data = [
            ['Metric', 'Best Checkpoint', 'Final Checkpoint'],
            ['Timestep', f"{timesteps[best_idx]:,}", f"{timesteps[-1]:,}"],
            ['Return %', f"{returns[best_idx]:+.2f}%", f"{returns[-1]:+.2f}%"],
            ['Profit Factor', f"{profit_factors[best_idx]:.2f}", f"{profit_factors[-1]:.2f}"],
            ['Trades', f"{trades[best_idx]}", f"{trades[-1]}"],
            ['Win Rate %', f"{win_rates[best_idx]:.1f}%", f"{win_rates[-1]:.1f}%"],
            ['Score', f"{scores[best_idx]:.2f}", f"{scores[-1]:.2f}"],
        ]
        
        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Выделяем заголовок
        for i in range(3):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
        
        # Выделяем лучшую колонку
        for i in range(1, len(summary_data)):
            table[(i, 1)].set_facecolor('#FFFACD')
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = model_dir / 'checkpoints_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Checkpoint comparison plot saved: {plot_path}")
    plt.close()


def create_checkpoints_markdown(metrics_list: list, model_dir: Path, best_checkpoint_name: str):
    """
    Создает markdown файл с таблицей сравнения чекпоинтов
    
    Args:
        metrics_list: Список метрик всех чекпоинтов
        model_dir: Путь к директории модели
        best_checkpoint_name: Имя выбранного лучшего чекпоинта
    """
    if not metrics_list:
        print("⚠️  No checkpoint metrics available for markdown")
        return
    
    # Сортируем по timestep
    metrics_list = sorted(metrics_list, key=lambda x: x.get('timestep', 0))

    thresholds = load_thresholds_from_config(model_dir)
    plateau = compute_plateau(metrics_list, best_checkpoint_name, thresholds)
    live_verdict = compute_live_verdict(metrics_list, best_checkpoint_name, thresholds, plateau)
    
    md_path = model_dir / 'CHECKPOINTS_COMPARISON.md'
    
    # Загружаем config.json для отображения параметров обучения
    config_path = model_dir / 'config.json'
    config_data = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as cf:
            config_data = json.load(cf)
    
    # Загружаем информацию о min_trades из selected_best_by_metrics.json
    min_trades_info = None
    selector_config_path = model_dir / "selected_best_by_metrics.json"
    if selector_config_path.exists():
        with open(selector_config_path, 'r', encoding='utf-8') as sf:
            selector_data = json.load(sf)
            min_trades_info = {
                'min_trades': selector_data.get('min_trades', 'N/A'),
                'timeframe': selector_data.get('min_trades_info', {}).get('timeframe', 'N/A'),
                'test_days': selector_data.get('min_trades_info', {}).get('test_days', 'N/A'),
                'test_bars': selector_data.get('min_trades_info', {}).get('test_bars', 'N/A')
            }
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Checkpoints Comparison\n\n")
        f.write(f"**Model:** `{model_dir.name}`\n\n")
        
        # Добавляем имя конфиг-файла, если оно есть
        if config_data and config_data.get('config_file'):
            config_file = config_data.get('config_file')
            f.write(f"**Config File:** `{config_file}`\n\n")
        
        f.write(f"**Best Checkpoint:** `{best_checkpoint_name}` ⭐\n\n")
        
        # Добавляем информацию о минимальном количестве сделок
        if min_trades_info:
            f.write(f"**Min Trades Threshold:** {min_trades_info['min_trades']} ")
            f.write(f"(calculated for {min_trades_info['timeframe']} timeframe, ")
            f.write(f"{min_trades_info['test_days']} test days, ")
            f.write(f"{min_trades_info['test_bars']} test bars)\n\n")
        
        # Добавляем информацию о параметрах обучения
        if config_data:
            f.write("## Training Configuration\n\n")
            
            # Основные параметры
            f.write("### General Parameters\n\n")
            f.write(f"- **Symbol:** {config_data.get('symbol', 'N/A')}\n")
            f.write(f"- **Timeframe:** {config_data.get('timeframe', 'N/A')}\n")
            f.write(f"- **Algorithm:** {config_data.get('algorithm', 'N/A')}\n")
            f.write(f"- **Total Timesteps:** {config_data.get('total_timesteps', 0):,}\n")
            f.write(f"- **Training Days:** {config_data.get('days', 0)}\n")
            f.write(f"- **Test Days:** {config_data.get('test_days_target', 0)}\n")
            f.write(f"- **Lookback:** {config_data.get('lookback', 0)}\n\n")
            
            # Гиперпараметры
            hyper = config_data.get('hyperparameters', {})
            if hyper:
                f.write("### Hyperparameters\n\n")
                f.write(f"- **Learning Rate:** {hyper.get('learning_rate', 'N/A')}\n")
                f.write(f"- **n_steps:** {hyper.get('n_steps', 'N/A')}\n")
                f.write(f"- **batch_size:** {hyper.get('batch_size', 'N/A')}\n")
                f.write(f"- **n_epochs:** {hyper.get('n_epochs', 'N/A')}\n")
                f.write(f"- **gamma:** {hyper.get('gamma', 'N/A')}\n")
                f.write(f"- **gae_lambda:** {hyper.get('gae_lambda', 'N/A')}\n")
                f.write(f"- **clip_range:** {hyper.get('clip_range', 'N/A')}\n")
                f.write(f"- **ent_coef:** {hyper.get('ent_coef', 'N/A')}\n\n")
            
            # Результаты обучения
            eval_res = config_data.get('evaluation_results', {})
            if eval_res:
                f.write("### Training Results\n\n")
                training_duration = eval_res.get('training_duration_minutes', 0)
                f.write(f"- **Training Duration:** {training_duration:.1f} minutes ({training_duration/60:.2f} hours)\n")
                f.write(f"- **Final Return:** {eval_res.get('final_return_pct', 0):+.2f}%\n")
                f.write(f"- **Profit Factor:** {eval_res.get('profit_factor', 0):.2f}\n")
                f.write(f"- **Max Drawdown:** {eval_res.get('max_drawdown_pct', 0):.2f}%\n")
                f.write(f"- **Win Rate:** {eval_res.get('win_rate_pct', 0):.1f}%\n")
                f.write(f"- **Total Trades:** {eval_res.get('total_trades', 0)}\n")
                f.write(f"- **Sharpe Ratio:** {eval_res.get('sharpe_ratio', 0):.2f}\n\n")
        
        f.write("---\n\n")
        
        # Таблица со всеми чекпоинтами
        f.write("## All Checkpoints Performance\n\n")
        f.write("| Step | Episodes | Return % | PF | Trades | Win % | Max DD % | Sharpe | Score | Status |\n")
        f.write("|------|----------|----------|----|----|-------|----------|--------|-------|--------|\n")
        
        for m in metrics_list:
            checkpoint = m.get('checkpoint', 'N/A')
            step = m.get('timestep', 0)
            episodes = m.get('episodes', 0)
            ret = m.get('total_return_pct', 0)
            pf = m.get('profit_factor', 0)
            trades = m.get('total_trades', 0)
            wr = m.get('win_rate_pct', 0)
            dd = m.get('max_drawdown_pct', 0)
            sharpe = m.get('sharpe_ratio', 0)
            score = m.get('score', 0)
            
            # Определяем статус
            is_best = best_checkpoint_name in checkpoint
            status = "⭐ **BEST**" if is_best else ""
            
            # Форматируем строку
            episodes_str = f"{episodes:.1f}" if episodes > 0 else "-"
            
            f.write(f"| {step:,} | {episodes_str} | {ret:+.2f} | {pf:.2f} | {trades} | {wr:.1f} | {dd:.2f} | {sharpe:.2f} | {score:.2f} | {status} |\n")
        
        f.write("\n---\n\n")
        
        # Детальное сравнение Best vs Final
        best_metrics = next((m for m in metrics_list if best_checkpoint_name in m.get('checkpoint', '')), None)
        # Финальный чекпоинт - это последний чекпоинт с реальным timestep (не best/final)
        real_checkpoints = [m for m in metrics_list if m.get('timestep', 0) < 900000000]
        final_metrics = real_checkpoints[-1] if real_checkpoints else metrics_list[-1]
        
        if best_metrics:
            f.write("## Best vs Final Checkpoint\n\n")
            f.write("| Metric | Best Checkpoint ⭐ | Final Checkpoint | Difference |\n")
            f.write("|--------|-------------------|------------------|------------|\n")
            
            metrics_to_compare = [
                ('Timestep', 'timestep', ',d'),
                ('Return %', 'total_return_pct', '+.2f'),
                ('Profit Factor', 'profit_factor', '.2f'),
                ('Trades', 'total_trades', 'd'),
                ('Win Rate %', 'win_rate_pct', '.1f'),
                ('Max Drawdown %', 'max_drawdown_pct', '.2f'),
                ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
                ('Score', 'score', '.2f'),
            ]
            
            for label, key, fmt in metrics_to_compare:
                best_val = best_metrics.get(key, 0)
                final_val = final_metrics.get(key, 0)
                
                if 'd' in fmt:
                    diff = final_val - best_val
                    diff_str = f"{diff:+d}" if diff != 0 else "="
                    best_str = f"{best_val:{fmt}}"
                    final_str = f"{final_val:{fmt}}"
                else:
                    diff = final_val - best_val
                    diff_str = f"{diff:+.2f}" if abs(diff) > 0.01 else "="
                    best_str = f"{best_val:{fmt}}"
                    final_str = f"{final_val:{fmt}}"
                
                f.write(f"| {label} | {best_str} | {final_str} | {diff_str} |\n")
            
            f.write("\n---\n\n")
        
        # Анализ переобучения
        if best_metrics:
            best_return = best_metrics.get('total_return_pct', 0)
            final_return = final_metrics.get('total_return_pct', 0)
            overfitting = best_return - final_return
            
            f.write("## Overfitting Analysis\n\n")
            f.write(f"- **Best Model Return:** {best_return:+.2f}%\n")
            f.write(f"- **Final Model Return:** {final_return:+.2f}%\n")
            f.write(f"- **Difference:** {overfitting:.2f}%\n\n")
            
            if abs(overfitting) < 3:
                f.write("✅ **Status:** Stable (minimal overfitting)\n\n")
            elif abs(overfitting) < 10:
                f.write("⚠️  **Status:** Moderate overfitting\n\n")
            else:
                f.write("❌ **Status:** Strong overfitting - consider reducing timesteps or adjusting hyperparameters\n\n")

        # Анализ плато / стабильности + вердикт пригодности
        f.write("## Plateau / Stability Analysis\n\n")
        f.write("This section checks whether performance forms a stable region (plateau) rather than an isolated spike.\n\n")
        if isinstance(plateau, dict) and plateau.get("best_score") is not None:
            f.write(f"- **Plateau criterion:** score ≥ best_score - eps\n")
            f.write(f"- **best_score:** {plateau.get('best_score', 0):+.2f}\n")
            f.write(f"- **eps used:** {plateau.get('eps_used', 0):.2f} (abs={thresholds.score_eps_abs:.2f}, rel={thresholds.score_eps_rel:.2f})\n")
            f.write(f"- **min plateau length:** {thresholds.plateau_min_len} consecutive checkpoints\n")
            f.write(f"- **valid checkpoints gate:** trades ≥ {thresholds.min_trades} AND episodes ≥ 2\n\n")

        plateau_len = plateau.get("len", 0) if isinstance(plateau, dict) else 0
        if isinstance(plateau, dict) and plateau.get("found", False):
            f.write(f"✅ **Plateau detected:** {plateau_len} checkpoints ({plateau.get('start_step', 'N/A'):,} → {plateau.get('end_step', 'N/A'):,})\n\n")
            f.write("| Step | Return % | PF | Trades | Max DD % | Score |\n")
            f.write("|------|----------|----|--------|----------|-------|\n")
            for p in plateau.get("steps", [])[:25]:
                f.write(
                    f"| {int(p.get('timestep', 0)):,} | {float(p.get('return_pct', 0)):+.2f} | {float(p.get('pf', 0)):.2f} | {int(p.get('trades', 0))} | {float(p.get('dd', 0)):.2f} | {float(p.get('score', 0)):+.2f} |\n"
                )
            if len(plateau.get("steps", [])) > 25:
                f.write(f"\n*Plateau list truncated to first 25 checkpoints (total: {len(plateau.get('steps', []))}).*\n\n")
            else:
                f.write("\n")
        else:
            min_len = plateau.get("min_len", thresholds.plateau_min_len) if isinstance(plateau, dict) else thresholds.plateau_min_len
            reason = plateau.get("reason", "plateau_not_found") if isinstance(plateau, dict) else "plateau_not_found"
            f.write(f"❌ **No stable plateau:** longest region below threshold is {plateau_len} checkpoints (min {min_len}).\n")
            f.write(f"- **Reason:** {reason}\n\n")

        f.write("## Live Readiness Verdict\n\n")
        verdict_status = live_verdict.get('status', 'FAIL')
        if verdict_status == 'PASS':
            f.write("✅ **PASS:** model looks stable enough to proceed to paper/forward testing.\n\n")
        else:
            f.write("❌ **FAIL:** do NOT treat this model as trade-ready based on checkpoint backtests.\n\n")

        f.write("**Decision thresholds (defaults or config.json overrides):**\n\n")
        th = live_verdict.get('thresholds', {})
        f.write(f"- min_trades: {th.get('min_trades')}\n")
        f.write(f"- pf_min: {th.get('pf_min')}\n")
        f.write(f"- dd_max: {th.get('dd_max')}\n")
        f.write(f"- min_return_pct: {th.get('min_return_pct')}\n")
        f.write(f"- plateau_min_len: {th.get('plateau_min_len')}\n")
        f.write(f"- tail_len: {th.get('tail_len')}\n")
        f.write(f"- tail_min_profitable_ratio: {th.get('tail_min_profitable_ratio')}\n\n")

        reasons = live_verdict.get('reasons', [])
        if reasons:
            f.write("**Reasons:**\n\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # Сравнение со специальными моделями (best_model.zip, final_model.zip)
        special_models = [m for m in metrics_list if m.get('timestep', 0) >= 900000000]
        if special_models:
            f.write("## Comparison with Special Models\n\n")
            f.write("*These models were NOT included in the selection process. Shown for reference only.*\n\n")
            f.write("| Model | Description | Return % | PF | Trades | Win % | Score |\n")
            f.write("|-------|-------------|----------|----|----|-------|-------|\n")
            
            for sm in special_models:
                model_name = sm.get('checkpoint', 'N/A')
                ret = sm.get('total_return_pct', 0)
                pf = sm.get('profit_factor', 0)
                trades = sm.get('total_trades', 0)
                wr = sm.get('win_rate_pct', 0)
                score = sm.get('score', 0)
                
                if 'best' in model_name.lower():
                    desc = "Saved by Stable-Baselines3 during training (best validation reward)"
                elif 'final' in model_name.lower():
                    desc = "Last checkpoint at end of training"
                else:
                    desc = "Special model"
                
                f.write(f"| {model_name} | {desc} | {ret:+.2f} | {pf:.2f} | {trades} | {wr:.1f} | {score:.2f} |\n")
            
            # Сравнение лучшего регулярного чекпоинта с best_model.zip
            if best_metrics:
                best_model = next((m for m in special_models if 'best' in m.get('checkpoint', '').lower()), None)
                if best_model:
                    f.write("\n### Selected Checkpoint vs Stable-Baselines3 Best Model\n\n")
                    best_return = best_metrics.get('total_return_pct', 0)
                    sb3_return = best_model.get('total_return_pct', 0)
                    diff = best_return - sb3_return
                    
                    f.write(f"- **Selected Checkpoint Return:** {best_return:+.2f}%\n")
                    f.write(f"- **SB3 Best Model Return:** {sb3_return:+.2f}%\n")
                    f.write(f"- **Difference:** {diff:+.2f}%\n\n")
                    
                    if diff > 0:
                        f.write("✅ **Selected checkpoint outperforms SB3's best model on trading metrics**\n\n")
                    elif diff < -1:
                        f.write("⚠️  **SB3's best model outperforms selected checkpoint**\n\n")
                    else:
                        f.write("➡️  **Similar performance**\n\n")
            
            f.write("---\n\n")
        
        f.write("*Generated automatically after training completion*\n")
    
    print(f"📝 Checkpoint comparison markdown saved: {md_path}")


def train_agent(args):
    """Основная функция обучения."""
    
    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        import torch
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print(f"\n{'='*70}")
    print(f"🚀 Training RL Agent v2")
    print(f"{'='*70}")
    print(f"📊 Symbol: {args.symbol}")
    print(f"⏰ Timeframe: {args.timeframe}")
    print(f"🤖 Algorithm: {args.algorithm}")
    print(f"📈 Days: {args.days}")
    print(f"💰 Initial Balance: ${args.initial_balance:,.0f}")
    print(f"🎯 Total Timesteps: {args.total_timesteps:,}")
    
    # Проверяем доступность GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 Device: GPU ({gpu_name})")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print(f"💻 Device: CPU (GPU not available)")
        print(f"   ⚠️  Training will be slower on CPU")
    
    print(f"{'='*70}\n")
    
    # Загружаем данные
    print("📊 Loading data...")
    loader = DataLoader(cache_dir="data/cache")
    
    # Если флаг --no-cache, сначала очищаем кэш
    if args.no_cache:
        print(f"🗑️  Clearing cache for {args.symbol}...")
        loader.clear_cache(args.symbol)
    
    df = loader.load_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        use_cache=not args.no_cache,
        max_cache_age_hours=args.max_cache_age
    )
    print(f"✅ Loaded {len(df)} bars")

    # Проверяем, что загруженных данных достаточно для запрошенного периода
    if len(df) > 0:
        first_ts = df['timestamp'].iloc[0]
        last_ts = df['timestamp'].iloc[-1]
        actual_days = (last_ts - first_ts).days
        required_days = args.days
        threshold = required_days * 0.9  # допуск 10%

        print(f"📅 Requested: {required_days} days | Available: {actual_days} days "
              f"({first_ts.strftime('%Y-%m-%d')} → {last_ts.strftime('%Y-%m-%d')})")

        if actual_days < threshold:
            print(f"\n❌ ОШИБКА: Недостаточно данных для обучения!")
            print(f"   Символ:    {args.symbol} ({args.timeframe})")
            print(f"   Запрошено: {required_days} дней")
            print(f"   Доступно:  {actual_days} дней (с {first_ts.strftime('%Y-%m-%d')})")
            print(f"   💡 Установите 'days' ≤ {actual_days} в конфиге: {args.config}")
            sys.exit(1)

    # Рассчитываем features (ENHANCED VERSION)
    print("🔧 Calculating features (enhanced with volume profile & order flow)...")
    engineer = FeatureEngineer()
    df = engineer.calculate_features(df)
    print(f"✅ Created {len(engineer.get_feature_names(df))} features")
    
    # Опционально добавляем Multi-Timeframe features (если флаг --use-mtf)
    if args.use_mtf:
        print("🔧 Adding Multi-Timeframe features (Weekly/Monthly)...")
        df = engineer.add_multi_timeframe_features(df)
        print(f"✅ Total features: {len(engineer.get_feature_names(df))}")
    else:
        print("ℹ️  Multi-Timeframe features disabled (use --use-mtf to enable)")
    
    # Нормализация (важно для нейронных сетей)
    print("📊 Normalizing features...")
    df = engineer.normalize_features(df, method='zscore')
    
    # Разделение на train/test
    # Если задан test_days_target, сначала оценим фактическую длину данных в днях и пересчитаем train_ratio
    if args.test_days_target is not None:
        if hasattr(df.index[0], 'to_pydatetime'):
            data_start = df.index[0]
            data_end = df.index[-1]
        else:
            data_start = df['timestamp'].iloc[0]
            data_end = df['timestamp'].iloc[-1]

        total_days = (data_end - data_start).days if hasattr(data_end, 'days') else int((data_end.timestamp() - data_start.timestamp()) / 86400)
        # Желаемая длина теста не может быть больше общего периода
        target_test_days = min(max(int(args.test_days_target), 1), max(total_days - 1, 1))
        train_days = max(total_days - target_test_days, 1)
        auto_train_ratio = train_days / max(total_days, 1)
        # Ограничиваем разумный диапазон для обучения
        auto_train_ratio = max(0.5, min(auto_train_ratio, 0.99))
        args.train_ratio = auto_train_ratio
        print(f"🔧 Using test_days_target={args.test_days_target}d → total_days={total_days}d → train_ratio={args.train_ratio:.4f}")

    split_idx = int(len(df) * args.train_ratio)
    print(f"🔧 Using train_ratio={args.train_ratio}: split at index {split_idx}")
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"✅ Train: {len(train_df)} bars, Test: {len(test_df)} bars")
    
    # Сохраняем даты периодов (до потери индекса)
    train_start = train_df.index[0] if hasattr(train_df.index[0], 'strftime') else df['timestamp'].iloc[0]
    train_end = train_df.index[-1] if hasattr(train_df.index[-1], 'strftime') else df['timestamp'].iloc[split_idx - 1]
    test_start = test_df.index[0] if hasattr(test_df.index[0], 'strftime') else df['timestamp'].iloc[split_idx]
    test_end = test_df.index[-1] if hasattr(test_df.index[-1], 'strftime') else df['timestamp'].iloc[-1]
    
    train_days = (train_end - train_start).days if hasattr(train_end, 'days') else int((train_end.timestamp() - train_start.timestamp()) / 86400)
    test_days = (test_end - test_start).days if hasattr(test_end, 'days') else int((test_end.timestamp() - test_start.timestamp()) / 86400)
    
    # Показываем периоды данных
    print(f"🗓  Data periods:")
    print(f"   Train: {train_start.strftime('%Y-%m-%d %H:%M:%S')} → {train_end.strftime('%Y-%m-%d %H:%M:%S')}  ({train_days} days, {len(train_df)} bars)")
    print(f"   Test : {test_start.strftime('%Y-%m-%d %H:%M:%S')} → {test_end.strftime('%Y-%m-%d %H:%M:%S')}  ({test_days} days, {len(test_df)} bars)")
    
    # Проверка актуальности данных
    current_date = datetime.now()
    hours_since_last_data = (current_date - test_end).total_seconds() / 3600 if hasattr(test_end, 'timestamp') else 0
    if hours_since_last_data > 48:
        print(f"   ⚠️  WARNING: Test data ends {hours_since_last_data:.1f}h ago! Data may be outdated.")
        print(f"   💡 Consider clearing cache: loader.clear_cache('{args.symbol}')")
    
    # Генерируем имя модели (ПОСЛЕ загрузки данных, чтобы знать test_days)
    if args.model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Формат как у XGBoost: SYMBOL_TF_ALGO_TRAINd_BTTESTd_TIMESTAMP
        args.model_name = f"{args.symbol}_{args.timeframe}_{args.algorithm}_{args.days}d_bt{test_days}d_{timestamp}"
    
    # Создаем директории
    model_dir = Path(f"rl_system/models/{args.model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем файл для лога обучения в Markdown
    training_log_md = model_dir / 'training_log.md'
    
    # Инициализируем Markdown лог
    with open(training_log_md, 'w', encoding='utf-8') as f:
        f.write(f"# Training Log: {args.model_name}\n\n")
        f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Symbol:** {args.symbol}\n")
        f.write(f"- **Timeframe:** {args.timeframe}\n")
        f.write(f"- **Algorithm:** {args.algorithm}\n")
        f.write(f"- **Total Timesteps:** {args.total_timesteps:,}\n")
        f.write(f"- **Learning Rate:** {args.learning_rate if args.learning_rate else 'default'}\n")
        f.write(f"- **n_steps:** {args.n_steps if args.n_steps else 'default'}\n")
        f.write(f"- **Training Days:** {args.days}\n")
        f.write(f"- **Test Days:** {test_days}\n")
        f.write(f"- **Initial Balance:** ${args.initial_balance:,.2f}\n")
        f.write(f"- **Position Size:** {args.position_size*100:.1f}%\n")
        f.write(f"- **Commission:** {args.commission*100:.2f}%\n\n")
        f.write(f"---\n\n")
        f.write(f"## Training Progress\n\n")
    
    # Автоматический расчет eval_freq (если не задан вручную)
    if args.eval_freq == 50000:  # Значение по умолчанию
        # Evaluation после каждого episode
        auto_eval_freq = len(train_df)
        print(f"🔧 Auto eval_freq: {auto_eval_freq} (= 1 episode)")
        args.eval_freq = auto_eval_freq
    else:
        print(f"🔧 Manual eval_freq: {args.eval_freq}")
    
    # Логируем используемые периоды train/test
    print("🗓  Data periods:")
    print(f"   Train: {train_start.strftime('%Y-%m-%d %H:%M:%S')} → {train_end.strftime('%Y-%m-%d %H:%M:%S')}  ({train_days} days, {len(train_df)} bars)")
    print(f"   Test : {test_start.strftime('%Y-%m-%d %H:%M:%S')} → {test_end.strftime('%Y-%m-%d %H:%M:%S')}  ({test_days} days, {len(test_df)} bars)")

    # Создаем окружения
    print("🎮 Creating environments...")
    train_env = create_env(train_df, args, is_eval=False)
    eval_env = create_env(test_df, args, is_eval=True)
    
    # Получаем оптимальные гиперпараметры для алгоритма
    defaults = get_algorithm_defaults(args.algorithm)
    
    # Override defaults с параметрами командной строки
    if args.learning_rate is not None:
        defaults['learning_rate'] = args.learning_rate
    if args.ent_coef is not None:
        defaults['ent_coef'] = args.ent_coef
    if args.n_steps is not None:
        defaults['n_steps'] = args.n_steps
    if args.batch_size is not None:
        defaults['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        defaults['n_epochs'] = args.n_epochs
    if args.gamma is not None:
        defaults['gamma'] = args.gamma
    if args.clip_range is not None:
        defaults['clip_range'] = args.clip_range
    
    # Создаем модель
    print(f"🤖 Creating {args.algorithm} model...")
    model = create_model(args.algorithm, train_env, args, defaults)
    
    # Сохраняем конфигурацию
    config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'algorithm': args.algorithm,
        'days': args.days,
        'train_ratio': args.train_ratio,
        'test_days_target': args.test_days_target,
        'config_file': args.config if hasattr(args, 'config') and args.config else None,
        'initial_balance': args.initial_balance,
        'commission': args.commission,
        'slippage': args.slippage,
        'lookback': args.lookback,
        'position_size': args.position_size,
        'enable_short': args.enable_short,
        'stop_loss_pct': args.stop_loss,
        'max_holding_bars': args.max_holding_bars,
        'hold_flat_penalty': args.hold_flat_penalty,
        'min_trades_target': args.min_trades_target,
        'too_few_trades_penalty': args.too_few_trades_penalty,
        'total_timesteps': args.total_timesteps,
        'hyperparameters': defaults,
        'n_features': len(engineer.get_feature_names(df)),
        'train_size': len(train_df),
        'test_size': len(test_df),
        # Даты периодов обучения и тестирования
        'train_period': {
            'start': train_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end': train_end.strftime('%Y-%m-%d %H:%M:%S'),
            'days': train_days
        },
        'test_period': {
            'start': test_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end': test_end.strftime('%Y-%m-%d %H:%M:%S'),
            'days': test_days
        },
    }

    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(model_dir / 'checkpoints'),
        name_prefix='rl_model'
    )
    
    # Создаем training log callback
    # По умолчанию логируем каждые n_steps (как консоль)
    # Можно изменить частоту через log_interval_multiplier (1=каждый rollout, 10=каждые 10 rollouts)
    training_log_callback = TrainingLogCallback(
        log_file=training_log_md,
        n_steps=defaults.get('n_steps', 32),
        log_interval_multiplier=1  # 1 = логируем так же часто как консоль
    )
    
    # Early stopping callback (optional)
    if args.early_stopping:
        print(f"⏹️  Early stopping enabled (patience={args.patience} evaluations)")
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.patience,
            min_evals=5,  # Minimum evals before stopping
            verbose=1
        )
    else:
        stop_callback = None
    
    # Используем модифицированный eval callback с логированием
    eval_callback = DetailedEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / 'best'),
        log_path=str(model_dir / 'eval_logs'),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,  # Pass stop_callback here
        log_file=training_log_md  # Передаем файл лога
    )
    
    # Объединяем все callbacks
    callback = CallbackList([checkpoint_callback, training_log_callback, eval_callback])
    
    # Обучение
    print(f"\n🚀 Starting training ({args.algorithm})...\n")
    print(f"📝 Training log will be saved to: {training_log_md}")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Завершаем лог обучения
    with open(training_log_md, 'a', encoding='utf-8') as f:
        f.write(f"\n## Training Completed\n\n")
        f.write(f"**Finished:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Duration:** {duration/60:.1f} minutes ({duration/3600:.2f} hours)\n")
        f.write(f"**Total Timesteps Completed:** {args.total_timesteps:,}\n\n")
    
    print(f"\n✅ Training log saved to: {training_log_md}")
    
    # Сохраняем финальную модель
    final_model_path = model_dir / 'final_model.zip'
    model.save(str(final_model_path))
    
    # Оцениваем FINAL модель (для сравнения с BEST)
    print(f"\n📊 Running evaluation on FINAL model (last training step)...")
    final_eval_env = CryptoTradingEnv(
        df=test_df,
        initial_balance=10000.0,
        commission=config['commission'],
        slippage=config['slippage'],
        lookback_window=config['lookback'],
        position_size=config['position_size'],
        enable_short=config.get('enable_short', True),
        stop_loss_pct=config.get('stop_loss_pct', 0.0),
        max_holding_bars=config.get('max_holding_bars', 0),
        hold_flat_penalty=config.get('hold_flat_penalty', 0.00005),
        min_trades_target=config.get('min_trades_target', 0),
        too_few_trades_penalty=config.get('too_few_trades_penalty', 0.0)
    )
    
    obs, _ = final_eval_env.reset()
    final_episode_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = final_eval_env.step(action)
        final_episode_reward += reward
        done = terminated or truncated
    
    # Получаем метрики FINAL модели
    final_info = final_eval_env._get_info()
    
    # Сохраняем config с метриками FINAL модели
    config['model_type'] = 'final'
    config['evaluation_results'] = {
        'final_return_pct': round(final_info['total_return_pct'], 2),
        'final_balance': round(final_eval_env.balance, 2),
        'final_equity': round(final_eval_env.equity, 2),
        'total_trades': final_info['total_trades'],
        'winning_trades': final_info['winning_trades'],
        'losing_trades': final_info['losing_trades'],
        'win_rate_pct': round(final_info['win_rate'] * 100, 2),
        'profit_factor': round(final_info['profit_factor'], 2),
        'gross_profit': round(final_info['gross_profit'], 2),
        'gross_loss': round(final_info['gross_loss'], 2),
        'max_drawdown_pct': round(final_info['max_drawdown_pct'], 2),
        'sharpe_ratio': round(final_info['sharpe_ratio'], 2),
        'episode_reward': round(final_episode_reward, 2),
        'training_duration_minutes': round(duration/60, 1),
        'test_bars': len(test_df),
    }
    
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Сохраняем детальный лог сделок для FINAL модели (если есть)
    try:
        if getattr(final_eval_env, 'trades_log', None):
            trades = final_eval_env.trades_log
            trades_path = model_dir / 'final_trades_log.csv'
            fieldnames = list(trades[0].keys())
            with open(trades_path, 'w', newline='', encoding='utf-8') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
    except Exception as e:
        print(f"⚠️  Failed to save FINAL trades log: {e}")
    
    print(f"✅ FINAL model evaluation:")
    print(f"   Return: {final_info['total_return_pct']:+.2f}%")
    print(f"   Balance: ${final_eval_env.balance:,.2f} (started with ${args.initial_balance:,.2f})")
    print(f"   Episode Reward: {final_episode_reward:.2f}")
    print(f"   Trades: {final_info['total_trades']} (Win Rate: {final_info['win_rate']*100:.1f}%)")
    print(f"   Profit Factor: {final_info['profit_factor']:.2f}")
    
    # Оцениваем BEST модель, если она существует
    best_model_path = model_dir / 'best' / 'best_model.zip'
    if best_model_path.exists():
        print(f"\n📊 Running evaluation on BEST model...")
        
        # Загружаем best модель
        algorithm_classes = {'PPO': PPO, 'A2C': A2C, 'SAC': SAC, 'TD3': TD3}
        model_class = algorithm_classes.get(config['algorithm'])
        best_model = model_class.load(str(best_model_path))
        
        # Создаем чистое окружение для оценки
        best_eval_env = CryptoTradingEnv(
            df=test_df,
            initial_balance=10000.0,
            commission=config['commission'],
            slippage=config['slippage'],
            lookback_window=config['lookback'],
            position_size=config['position_size'],
            enable_short=config.get('enable_short', True),
            stop_loss_pct=config.get('stop_loss_pct', 0.0),
            max_holding_bars=config.get('max_holding_bars', 0),
            hold_flat_penalty=config.get('hold_flat_penalty', 0.00005),
            min_trades_target=config.get('min_trades_target', 0),
            too_few_trades_penalty=config.get('too_few_trades_penalty', 0.0)
        )
        
        # Запускаем оценку
        obs, _ = best_eval_env.reset()
        best_episode_reward = 0
        done = False
        
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = best_eval_env.step(action)
            best_episode_reward += reward
            done = terminated or truncated
        
        # Получаем метрики best модели
        best_info = best_eval_env._get_info()
        
        # Сохраняем config для best модели
        best_config = config.copy()
        best_config['model_type'] = 'best'
        best_config['evaluation_results'] = {
            'final_return_pct': round(best_info['total_return_pct'], 2),
            'final_balance': round(best_eval_env.balance, 2),
            'final_equity': round(best_eval_env.equity, 2),
            'total_trades': best_info['total_trades'],
            'winning_trades': best_info['winning_trades'],
            'losing_trades': best_info['losing_trades'],
            'win_rate_pct': round(best_info['win_rate'] * 100, 2),
            'profit_factor': round(best_info['profit_factor'], 2),
            'gross_profit': round(best_info['gross_profit'], 2),
            'gross_loss': round(best_info['gross_loss'], 2),
            'max_drawdown_pct': round(best_info['max_drawdown_pct'], 2),
            'sharpe_ratio': round(best_info['sharpe_ratio'], 2),
            'episode_reward': round(best_episode_reward, 2),
            'training_duration_minutes': round(duration/60, 1),
            'test_bars': len(test_df),
        }
        
        best_config_path = model_dir / 'best' / 'config.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=4)

        # Сохраняем детальный лог сделок для BEST модели (если есть)
        try:
            if getattr(best_eval_env, 'trades_log', None):
                trades = best_eval_env.trades_log
                trades_path = model_dir / 'best' / 'best_trades_log.csv'
                fieldnames = list(trades[0].keys())
                with open(trades_path, 'w', newline='', encoding='utf-8') as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(trades)
        except Exception as e:
            print(f"⚠️  Failed to save BEST trades log: {e}")
        
        # Сохраняем историю eval (траекторию обучения)
        eval_history_path = model_dir / 'eval_history.json'
        with open(eval_history_path, 'w') as f:
            json.dump(eval_callback.eval_history, f, indent=4)
        
        print(f"✅ BEST model evaluation:")
        print(f"   Return: {best_info['total_return_pct']:+.2f}%")
        print(f"   Trades: {best_info['total_trades']} (Win Rate: {best_info['win_rate']*100:.1f}%)")
        print(f"   Profit Factor: {best_info['profit_factor']:.2f}")
        
        # Сравнение FINAL vs BEST для диагностики переобучения
        return_diff = abs(best_info['total_return_pct'] - final_info['total_return_pct'])
        print(f"\n📊 Overfitting Analysis:")
        print(f"   FINAL model: {final_info['total_return_pct']:+.2f}% (step {args.total_timesteps})")
        print(f"   BEST model:  {best_info['total_return_pct']:+.2f}% (saved during training)")
        print(f"   Difference:  {return_diff:.2f}%", end="")
        
        if return_diff < 3:
            print(" ✅ (stable, minimal overfitting)")
        elif return_diff < 10:
            print(" ⚠️  (moderate overfitting)")
        else:
            print(" ❌ (strong overfitting - consider reducing timesteps)")
    else:
        print(f"\n⚠️  BEST model not found (training might have been interrupted)")
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"{'='*70}")
    print(f"⏱️  Duration: {duration/60:.1f} minutes ({duration:.0f} seconds)")
    print(f"💾 Models saved:")
    print(f"   BEST model (use for production): {model_dir / 'best' / 'best_model.zip'} ⭐")
    print(f"   BEST config: {model_dir / 'best' / 'config.json'}")
    print(f"   FINAL model (for comparison): {final_model_path}")
    print(f"   FINAL config: {model_dir / 'config.json'}")
    print(f"{'='*70}\n")
    print(f"🎉 Success! Your {args.algorithm} agent is ready to trade!")
    print(f"\n💡 Next steps:")
    print(f"   1. Check BEST model performance: cat {model_dir / 'best' / 'config.json'}")
    print(f"   2. Detailed evaluation: python rl_system/evaluate_agent.py --model-path {model_dir}")
    print(f"   3. View training logs: tensorboard --logdir rl_system/logs/{args.model_name}/tensorboard")
    print(f"   4. Compare with other algorithms\n")

    # Автоматический запуск селектора лучшего чекпоинта по торговым метрикам
    try:
        print(f"\n🔍 Running checkpoint selector by trading metrics with dynamic min_trades...")
        
        # Вызываем селектор с динамическим расчетом min_trades (None = auto)
        # Расчет происходит внутри select_best_checkpoint на основе timeframe и длины бэктеста
        metrics_list = select_best_checkpoint(model_dir, min_trades=None, train_size=len(train_df))
        
        # Загружаем информацию о выбранном лучшем чекпоинте
        selector_config_path = model_dir / "selected_best_by_metrics.json"
        if selector_config_path.exists():
            with open(selector_config_path, 'r') as f:
                best_selection = json.load(f)
                # Поддерживаем старый и новый формат
                if "best_checkpoint" in best_selection:
                    best_checkpoint_name = best_selection["best_checkpoint"].get('checkpoint', '')
                else:
                    # Старый формат
                    best_checkpoint_name = best_selection.get('checkpoint', '')
            
            print("✅ Selector finished; saved selected_best_by_metrics.json")
            
            # Создаем визуализацию и markdown
            if metrics_list:
                print(f"\n📊 Creating checkpoint comparison visualizations...")
                plot_checkpoints_comparison(metrics_list, model_dir, best_checkpoint_name)
                create_checkpoints_markdown(metrics_list, model_dir, best_checkpoint_name)
                print("✅ Visualizations created successfully")
                
                # Выполняем комплексную оценку качества модели
                print(f"\n🎯 Performing model quality assessment...")
                try:
                    assessment = assess_model_quality(metrics_list, model_dir, best_checkpoint_name)
                    
                    # Добавляем результаты оценки в markdown
                    md_path = model_dir / 'CHECKPOINTS_COMPARISON.md'
                    quality_md = format_quality_assessment_markdown(assessment)
                    
                    with open(md_path, 'a', encoding='utf-8') as f:
                        f.write(quality_md)
                    
                    print(f"✅ Quality assessment completed:")
                    print(f"   Verdict: {assessment['verdict']}")
                    print(f"   Confidence: {assessment['confidence']}%")
                    print(f"   Overall Score: {assessment['overall_score_pct']:.1f}%")
                except Exception as e:
                    print(f"⚠️  Quality assessment failed: {e}")
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        print(f"⚠️  Selector/visualization failed: {e}")
        import traceback
        traceback.print_exc()

    # Автоматическое обновление таблицы моделей
    try:
        print(f"\n📝 Auto-updating models table for {args.symbol}...")
        import subprocess
        
        # Получаем путь к Python интерпретатору
        python_exe = sys.executable
        
        # Запускаем generate_models_md_table.py
        result = subprocess.run(
            [python_exe, 'rl_system/generate_models_md_table.py', '--symbol', args.symbol],
            cwd=Path(__file__).parent.parent,  # project root directory
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Models table updated successfully: RL_{args.symbol.upper()}_MODELS_TABLE.md")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"⚠️  Models table update failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print(f"⚠️  Models table update timed out after 60 seconds")
    except Exception as e:
        print(f"⚠️  Models table update failed: {e}")
        import traceback
        traceback.print_exc()

    # Отправка уведомления в Telegram о завершении обучения
    if TELEGRAM_AVAILABLE and send_telegram_message:
        try:
            print(f"\n📱 Sending training completion notification to Telegram...")
            
            # Собираем информацию о лучшем чекпоинте
            best_checkpoint_name = None
            selector_config_path = model_dir / "selected_best_by_metrics.json"
            if selector_config_path.exists():
                with open(selector_config_path, 'r') as f:
                    best_selection = json.load(f)
                    # Поддерживаем старый и новый формат
                    if "best_checkpoint" in best_selection:
                        best_checkpoint_name = best_selection["best_checkpoint"].get('checkpoint', '')
                    else:
                        # Старый формат
                        best_checkpoint_name = best_selection.get('checkpoint', '')
            
            # Загружаем информацию о лучшей модели
            best_config_path = model_dir / 'best' / 'config.json'
            best_model_info = None
            if best_config_path.exists():
                with open(best_config_path, 'r') as f:
                    best_config = json.load(f)
                    if 'evaluation_results' in best_config:
                        eval_results = best_config['evaluation_results']
                        best_model_info = {
                            'total_return_pct': eval_results.get('final_return_pct', 0),
                            'win_rate': eval_results.get('win_rate_pct', 0) / 100,
                            'total_trades': eval_results.get('total_trades', 0),
                            'profit_factor': eval_results.get('profit_factor', 0),
                            'max_drawdown_pct': eval_results.get('max_drawdown_pct', 0),
                            'sharpe_ratio': eval_results.get('sharpe_ratio', 0),
                        }
            
            # Форматируем и отправляем сообщение
            message = format_training_completion_message(
                config=config,
                model_dir=model_dir,
                final_info=final_info,
                best_info=best_model_info,
                best_checkpoint_name=best_checkpoint_name,
                duration=duration,
                args=args
            )
            
            success = send_telegram_message(message)
            if success:
                print("✅ Telegram notification sent successfully")
            else:
                print("⚠️  Failed to send Telegram notification")
                
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  Telegram notifications are disabled (module not available)")


if __name__ == '__main__':
    args = parse_args()
    train_agent(args)
