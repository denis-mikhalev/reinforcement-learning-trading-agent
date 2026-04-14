"""
🤖 RL AGENT LIVE TRADING SYSTEM
================================

Запускает обученного RL агента в режиме реального времени:
- Непрерывный мониторинг цены
- Генерация сигналов BUY/SELL/HOLD
- Интеграция с Telegram (опционально)
- Сохранение сигналов в signals/

ИСПОЛЬЗОВАНИЕ:
    python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800 --continuous
    python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800 --continuous --telegram
"""

import argparse
import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import ccxt
try:
    import winsound
except ImportError:
    winsound = None

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_system.data_loader import DataLoader
from rl_system.feature_engineering import FeatureEngineer
from rl_system.trading_env import CryptoTradingEnv, Positions
from telegram_sender import send_trading_signal, format_price


class RLLiveTrader:
    """Live trading с RL агентом"""
    
    def __init__(self, model_path: str, symbol: str = None, timeframe: str = None, 
                 telegram_enabled: bool = False, continuous: bool = True, 
                 interval: int = None, no_sound: bool = False, signal_change_sound: bool = False):
        """
        Args:
            model_path: Путь к обученной модели
            symbol: Торговая пара (если None, берется из конфига)
            timeframe: Таймфрейм (если None, берется из конфига)
            telegram_enabled: Отправлять сигналы в Telegram
            continuous: Непрерывный мониторинг
            interval: Интервал проверки в секундах (по умолчанию зависит от таймфрейма)
        """
        self.model_path = Path(model_path)
        self.telegram_enabled = telegram_enabled
        self.continuous = continuous
        # Flag to suppress audible beeps
        self.no_sound = bool(no_sound)
        # Flag: beep only when signal changes (HOLD->BUY, BUY->SELL, etc.)
        self.signal_change_sound = bool(signal_change_sound)
        
        # Загружаем конфигурацию модели
        self.config = self._load_config()
        self.symbol = symbol or self.config.get('symbol', 'BTCUSDT')
        self.timeframe = timeframe or self.config.get('timeframe', '4h')
        
        # Устанавливаем интервал проверки
        self.interval = interval or self._default_interval()
        
        # Загружаем модель и нормализатор
        self.model, self.vec_normalize = self._load_model()
        
        # Инициализируем data loader и feature engineer
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()

        # Вспомогательное окружение, повторяющее логику обучения, для формирования obs
        self.env: Optional[CryptoTradingEnv] = None
        
        # Последний сигнал для предотвращения дублирования
        self.last_signal = None
        self.last_signal_time = None
        
        # Создаем директорию для сигналов
        self.signals_dir = Path(__file__).parent.parent / 'signals'
        self.signals_dir.mkdir(exist_ok=True)

        # Директория для хранения "последнего" сигнала каждой RL‑модели
        # Используется агрегатором для построения ансамблевой сводки.
        self.live_state_dir = Path(__file__).parent / 'live_state'
        self.live_state_dir.mkdir(exist_ok=True)
        
        # Загружаем метрики из best/config.json
        best_config_path = self.model_path / "best" / "config.json"
        best_config = {}
        if best_config_path.exists():
            with open(best_config_path, 'r') as f:
                best_config = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"🤖 RL LIVE TRADING SYSTEM INITIALIZED")
        print(f"{'='*70}")
        print(f"Model: {self.model_path.name}")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Algorithm: {best_config.get('algorithm', 'Unknown')}")
        print(f"Check Interval: {self._format_interval()}")
        print(f"Telegram: {'✅ ENABLED' if self.telegram_enabled else '❌ DISABLED'}")
        print(f"Mode: {'🔄 CONTINUOUS' if self.continuous else '🎯 SINGLE SHOT'}")
        # Detailed sound mode description
        if self.no_sound:
            sound_mode = "🔇 DISABLED"
        elif self.signal_change_sound:
            sound_mode = "🔔 SIGNAL CHANGE ONLY"
        else:
            sound_mode = "🔊 EVERY TRADE SIGNAL"
        print(f"Sound: {sound_mode}")
        
        # Показываем параметры окружения из конфига
        print(f"\n📋 Environment Parameters (from model config):")
        print(f"   Lookback: {self.config.get('lookback', 'N/A')}")
        print(f"   Features: {self.config.get('n_features', 'N/A')}")
        print(f"   Position Size: {self.config.get('position_size', 'N/A')}")
        print(f"   Commission: {self.config.get('commission', 'N/A')}")
        print(f"   Slippage: {self.config.get('slippage', 'N/A')}")
        print(f"   Enable Short: {self.config.get('enable_short', 'N/A')}")
        print(f"   Stop Loss %: {self.config.get('stop_loss_pct', 'N/A')}")
        
        # Показываем метрики качества модели (из best/config.json -> evaluation_results)
        eval_results = best_config.get('evaluation_results', {})
        if eval_results:
            print(f"\n📊 Model Performance (Backtest):")
            
            # Период бэктеста с точными датами
            train_period = best_config.get('train_period', {})
            test_period = best_config.get('test_period', {})
            
            print(f"   📅 Train: {train_period.get('start', 'N/A')} → {train_period.get('end', 'N/A')} ({train_period.get('days', 0)} days)")
            print(f"   📅 Test:  {test_period.get('start', 'N/A')} → {test_period.get('end', 'N/A')} ({test_period.get('days', 0)} days)")
            
            # Метрики производительности
            total_return = eval_results.get('final_return_pct', 0)
            quality_indicator = "✅" if total_return > 15 else "⚠️" if total_return > 5 else "❌"
            
            print(f"   {quality_indicator} Return: {total_return:+.2f}%")
            print(f"   Profit Factor: {eval_results.get('profit_factor', 0):.2f}")
            print(f"   Total Trades: {eval_results.get('total_trades', 0)}")
            print(f"   Win Rate: {eval_results.get('win_rate_pct', 0):.1f}%")
            print(f"   Sharpe Ratio: {eval_results.get('sharpe_ratio', 0):.2f}")
            
            gross_profit = eval_results.get('gross_profit', 0)
            gross_loss = eval_results.get('gross_loss', 0)
            print(f"   Gross P/L: ${gross_profit:+.2f} / ${gross_loss:.2f}")
        
        print(f"{'='*70}\n")
    
    def _load_config(self) -> Dict:
        """Загружает конфигурацию модели"""
        # Если model_path указывает на .zip файл, поднимаемся к директории модели
        if self.model_path.suffix == ".zip":
            # Для checkpoints/rl_model_XXX.zip нужно подняться на 2 уровня
            if self.model_path.parent.name == "checkpoints":
                model_dir = self.model_path.parent.parent
            else:
                model_dir = self.model_path.parent
        else:
            model_dir = self.model_path
        
        # Ищем config.json в директории модели
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        print(f"⚠️  Config not found at {config_path}, using defaults")
        return {}
    
    def _default_interval(self) -> int:
        """Возвращает интервал по умолчанию на основе таймфрейма"""
        tf_intervals = {
            '15m': 60,    # 1 минута (как XGBoost)
            '30m': 60,    # 1 минута (как XGBoost)
            '1h': 60,     # 1 минута (более частая проверка)
            '4h': 300,    # 5 минут (разумный баланс для 4h)
            '12h': 600,   # 10 минут
            '1d': 1800,   # 30 минут
        }
        return tf_intervals.get(self.timeframe, 300)

    def _timeframe_to_timedelta(self) -> timedelta:
        """Преобразует строковый таймфрейм ccxt ('6h', '12h', '1d', '15m', ...) в timedelta.

        Используется только для приблизительного расчета времени до/с момента закрытия свечи.
        """
        tf = str(self.timeframe)
        try:
            value = int(tf[:-1])
            unit = tf[-1]
        except Exception:
            # Fallback: считаем 1 бар = 5 минут
            return timedelta(minutes=5)

        if unit == 'm':
            return timedelta(minutes=value)
        if unit == 'h':
            return timedelta(hours=value)
        if unit == 'd':
            return timedelta(days=value)
        if unit == 'w':
            return timedelta(weeks=value)

        # Неизвестная единица – безопасный дефолт
        return timedelta(minutes=5)
    
    def _format_interval(self) -> str:
        """Форматирует интервал для вывода"""
        if self.interval < 60:
            return f"{self.interval} сек"
        elif self.interval < 3600:
            return f"{self.interval // 60} мин"
        else:
            return f"{self.interval // 3600} час"
    
    def _load_model(self) -> Tuple:
        """Загружает обученную модель и VecNormalize.

        Поддерживает оба варианта:
        - model_path указывает на директорию модели (старое поведение)
        - model_path указывает напрямую на .zip (конкретный чекпоинт)
        """
        # Определяем тип модели из названия (директории или файла)
        model_name = self.model_path.name
        algorithm = self.config.get('algorithm', 'PPO')

        if 'A2C' in model_name or algorithm == 'A2C':
            model_class = A2C
        else:
            model_class = PPO

        # Если model_path указывает прямо на .zip – используем его как есть
        if self.model_path.suffix == ".zip" and self.model_path.is_file():
            model_file = self.model_path
            # Базовая директория модели – родительская папка (ищем там config/vecnorm)
            model_dir = self.model_path.parent.parent if self.model_path.parent.name == "checkpoints" else self.model_path.parent
        else:
            # model_path – директория модели (старый формат)
            model_dir = self.model_path

            # Загружаем модель (пробуем разные названия внутри директории)
            model_file_candidates = [
                model_dir / "final_model.zip",
                model_dir / "best_model.zip",
                model_dir / "model.zip",
                model_dir / "best" / "best_model.zip",
            ]

            model_file = None
            for candidate in model_file_candidates:
                if candidate.exists():
                    model_file = candidate
                    break

            if model_file is None:
                raise FileNotFoundError(
                    f"Model file not found in {model_dir}. "
                    f"Tried: {[c.name for c in model_file_candidates]}"
                )

        model = model_class.load(str(model_file))

        # Загружаем VecNormalize если есть (привязан к директории модели)
        vec_normalize = None
        vecnorm_file = model_dir / "vec_normalize.pkl"
        if vecnorm_file.exists():
            vec_normalize = VecNormalize.load(str(vecnorm_file), DummyVecEnv([lambda: None]))

        print(f"✅ Loaded {algorithm} model from {model_file}")
        return model, vec_normalize
    
    def _fetch_latest_data(self, limit: int = 500) -> pd.DataFrame:
        """Получает последние данные с биржи"""
        try:
            # Конвертируем символ в формат с слэшем если нужно
            symbol_with_slash = self.symbol if '/' in self.symbol else f"{self.symbol[:-4]}/{self.symbol[-4:]}"
            
            # Пробуем разные биржи
            exchanges = [
                ('binance', ccxt.binance()),
                ('kucoin', ccxt.kucoin()),
                ('gate', ccxt.gate()),
            ]
            
            for exchange_name, exchange in exchanges:
                try:
                    exchange.load_markets()
                    
                    # Пробуем оба формата символа
                    for symbol_format in [symbol_with_slash, self.symbol]:
                        if symbol_format in exchange.markets:
                            ohlcv = exchange.fetch_ohlcv(symbol_format, self.timeframe, limit=limit)
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            # Подробный лог источника данных скрыт, чтобы не спамить консоль.
                            # При необходимости отладки можно раскомментировать строку ниже.
                            # print(f"✅ Fetched data from {exchange_name} ({symbol_format})")
                            return df
                except Exception as e:
                    print(f"⚠️ {exchange_name}: {e}")
                    continue
            
            raise ValueError(f"Could not fetch data for {self.symbol} or {symbol_with_slash}")
        
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def _prepare_observation(self, df: pd.DataFrame) -> np.ndarray:
        """Подготавливает observation через CryptoTradingEnv, как при обучении."""
        # Получаем все параметры из конфига модели (без дефолтов - они должны быть в конфиге)
        lookback = int(self.config.get('lookback', 64))
        initial_balance = float(self.config.get('initial_balance', 10000.0))
        commission = float(self.config.get('commission', 0.001))
        slippage = float(self.config.get('slippage', 0.0005))
        position_size = float(self.config.get('position_size', 0.3))
        enable_short = bool(self.config.get('enable_short', True))
        stop_loss_pct = float(self.config.get('stop_loss_pct', 0.0))
        max_holding_bars = int(self.config.get('max_holding_bars', 0))

        # Проверяем, нужны ли Multi-Timeframe фичи
        # 1. Пробуем взять из конфига
        use_mtf = self.config.get('use_mtf', None)
        if use_mtf is None:
            use_mtf = self.config.get('hyperparameters', {}).get('use_mtf', None)
        
        # 2. Если в конфиге нет - определяем по observation shape модели
        if use_mtf is None:
            # Временно создаем базовые фичи для проверки
            df_temp = self.feature_engineer.calculate_features(df.copy())
            expected_features = self.model.observation_space.shape[1]
            current_features = df_temp.shape[1] - 6  # Минус OHLCV + timestamp
            
            # Если ожидается больше фичей - значит использовались MTF (добавляют ~14 фичей)
            if expected_features > current_features:
                use_mtf = True
                # Детальный лог автоопределения MTF скрыт, чтобы не засорять live‑консоль.
                # print(f"🔍 Auto-detected MTF features: model expects {expected_features}, current {current_features}")
            else:
                use_mtf = False
        
        # 3. Добавляем MTF фичи ПЕРЕД calculate_features если нужно
        # (MTF требует DatetimeIndex, который есть в исходном df)
        if use_mtf:
            # Лог добавления MTF‑фич скрыт, чтобы не спамить консоль.
            # print("🔧 Adding Multi-Timeframe features (Weekly/Monthly)...")
            # Убедимся что timestamp это индекс, а не колонка
            if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            df = self.feature_engineer.add_multi_timeframe_features(df)
            # MTF возвращает df с timestamp как колонкой - это нормально
            # print(f"✅ MTF features added, total columns: {df.shape[1]}")
        
        # 4. Теперь строим все фичи (включая MTF если они были добавлены)
        df_proc = self.feature_engineer.calculate_features(df)

        # Логируем информацию о фичах: сколько и какие именно колонки идут в env
        feature_cols = [
            c for c in df_proc.columns
            if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ]
        # Подробные логи по фичам отключены, чтобы не засорять live‑консоль.
        # При отладке можно временно раскомментировать строки ниже.
        # print(f"✅ Total columns after feature engineering: {df_proc.shape[1]} (features: {len(feature_cols)})")
        # print(f"📋 Feature columns used for env (first 30): {feature_cols[:30]}")
        # if len(feature_cols) > 30:
        #     print(f"   ... and {len(feature_cols) - 30} more")
        
        # NOTE: для тренировки Multi‑TF фичи и нормализация применялись ДО создания env
        # Здесь они уже должны быть включены в df_proc, если модель так обучалась.

        # Проверяем, достаточно ли баров для lookback
        if len(df_proc) < lookback + 2:
            raise ValueError(f"Not enough data for env: need at least {lookback+2}, got {len(df_proc)}")

        # Если окружение ещё не создано или поменялась длина df, пересоздаём
        if self.env is None or getattr(self.env, 'lookback_window', None) != lookback:
            self.env = CryptoTradingEnv(
                df=df_proc,
                initial_balance=initial_balance,
                commission=commission,
                slippage=slippage,
                lookback_window=lookback,
                position_size=position_size,
                enable_short=enable_short,
                stop_loss_pct=stop_loss_pct,
                max_holding_bars=max_holding_bars,
            )
            print(f"📊 Environment created: lookback={lookback}, features={df_proc.shape[1]}, "
                f"commission={commission}, slippage={slippage}, position_size={position_size}")
            print(f"📊 Env observation space: {self.env.observation_space.shape}, "
                f"model expects: {self.model.observation_space.shape}")
        else:
            # Обновляем данные в существующем env
            self.env.df = df_proc.reset_index(drop=True)
            self.env.max_episode_steps = len(df_proc) - lookback - 1

        # Для live trading нам нужно observation с КОНЦА данных (текущая ситуация на рынке)
        # а не с начала (как при обучении)
        # Устанавливаем current_step на последнюю возможную позицию
        self.env.current_step = len(df_proc) - 1
        
        # Получаем observation с текущей позиции (последние lookback баров)
        obs = self.env._get_observation()

        # Нормализуем, если использовался VecNormalize при обучении
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)

        # Валидация размерности observation
        # Используем размерность из observation_space модели (надежнее чем конфиг)
        expected_shape = self.model.observation_space.shape
        if obs.shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch!\n"
                f"Expected: {expected_shape} (from model.observation_space)\n"
                f"Got: {obs.shape}\n"
                f"DataFrame has {df_proc.shape[1]} columns after feature engineering.\n"
                f"Env observation space: {self.env.observation_space.shape}\n"
                f"This usually means feature engineering differs between training and inference."
            )

        return obs
    
    def _action_to_signal(self, action: int) -> str:
        """Конвертирует действие агента в сигнал"""
        # Positions: HOLD=0, LONG=1, SHORT=2
        action_map = {
            0: 'HOLD',
            1: 'BUY',   # LONG
            2: 'SELL',  # SHORT
        }
        return action_map.get(action, 'HOLD')
    
    def _get_action_probabilities(self, obs: np.ndarray) -> Dict[str, float]:
        """Извлекает вероятности действий из policy"""
        try:
            import torch
            
            # obs уже в правильном формате и нормализован через VecNormalize
            # Конвертируем в tensor
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.as_tensor(obs).float()
            else:
                obs_tensor = obs
            
            # Добавляем batch dimension если нужно
            if len(obs_tensor.shape) == 2:  # (lookback, features)
                obs_tensor = obs_tensor.unsqueeze(0)  # (1, lookback, features)
            
            # Получаем распределение действий из policy
            with torch.no_grad():
                # Используем policy напрямую (obs уже нормализован)
                # Для A2C/PPO: policy.get_distribution возвращает distribution
                if hasattr(self.model.policy, 'get_distribution'):
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.cpu().numpy()[0]
                else:
                    # Fallback: извлекаем через features
                    features = self.model.policy.extract_features(obs_tensor)
                    if hasattr(self.model.policy, 'mlp_extractor'):
                        latent_pi, _ = self.model.policy.mlp_extractor(features)
                    else:
                        latent_pi = features
                    
                    # Получаем logits действий
                    action_logits = self.model.policy.action_net(latent_pi)
                    
                    # Применяем softmax для получения вероятностей
                    probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
            
            # Конвертируем в словарь
            return {
                'HOLD': float(probs[0]),
                'BUY': float(probs[1]),
                'SELL': float(probs[2]),
            }
        except Exception as e:
            # Логируем ошибку для отладки
            print(f"⚠️ Could not extract action probabilities: {e}")
            import traceback
            traceback.print_exc()
            # Если не удалось извлечь - возвращаем равные вероятности
            return {'HOLD': 0.33, 'BUY': 0.33, 'SELL': 0.33}
    
    def _calculate_position_size(self, current_price: float, atr: float, signal: str) -> Dict:
        """Рассчитывает размер позиции и SL/TP"""
        # Базовые множители для RL агентов (более широкие чем у XGBoost)
        if self.timeframe == '4h':
            sl_mult, tp_mult = 2.5, 3.0
        elif self.timeframe == '1h':
            sl_mult, tp_mult = 2.0, 2.5
        else:
            sl_mult, tp_mult = 2.0, 3.0
        
        if signal == 'BUY':
            stop_loss = current_price - (atr * sl_mult)
            take_profit = current_price + (atr * tp_mult)
        elif signal == 'SELL':
            stop_loss = current_price + (atr * sl_mult)
            take_profit = current_price - (atr * tp_mult)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_multiplier': sl_mult,
            'tp_multiplier': tp_mult,
        }
    
    def _get_model_info(self) -> Dict:
        """Возвращает информацию о модели для логов"""
        # Определяем удобочитаемый идентификатор модели (директория, в которой хранится модель)
        # Если пользователь передал .zip, пытаемся вернуть имя родительской папки модели
        try:
            if self.model_path.suffix == ".zip":
                # если файл лежит в checkpoints, поднимаемся на 2 уровня
                if self.model_path.parent.name == "checkpoints":
                    model_dir = self.model_path.parent.parent
                else:
                    model_dir = self.model_path.parent
            else:
                model_dir = self.model_path

            model_identifier = model_dir.name
        except Exception:
            model_identifier = self.model_path.name

        return {
            'model_identifier': model_identifier,  # e.g. BTCUSDT_12h_A2C_1095d_bt45d_20251127_233534
            'model_filename': self.model_path.name,  # e.g. rl_model_220000_steps.zip
            'algorithm': self.config.get('algorithm', 'Unknown'),
            'trained_on': {
                'symbol': self.config.get('symbol'),
                'timeframe': self.config.get('timeframe'),
                'training_days': self.config.get('training_days'),
                'timesteps': self.config.get('total_timesteps'),
            },
            'performance': {
                'final_return': self.config.get('final_return'),
                'sharpe_ratio': self.config.get('sharpe_ratio'),
                'max_drawdown': self.config.get('max_drawdown'),
            }
        }
    
    def _should_send_signal(self, signal: str, current_time: datetime) -> bool:
        """Проверяет, нужно ли отправлять сигнал.

        Обновлённая логика:
        - каждый новый BUY/SELL всегда уходит в Telegram;
        - HOLD по-прежнему не отправляем.

        Фильтрация по времени и по последнему сигналу отключена,
        чтобы избежать пропусков уведомлений при live‑запуске.
        """
        # Не отправляем HOLD, все остальные сигналы (BUY/SELL) отправляем всегда
        if signal == 'HOLD':
            return False
        return True
    
    def _save_signal_to_file(self, signal_data: Dict):
        """Сохраняет сигнал в файл"""
        # Пишем только отдельный лог на модель, чтобы статистика не смешивалась между моделями.

        # Отдельный файл сигналов под каждую модель (рекомендуемый вариант)
        model_info = signal_data.get('model_info', {}) if isinstance(signal_data, dict) else {}
        model_identifier = (
            model_info.get('model_identifier')
            or model_info.get('model_filename')
            or 'unknown_model'
        )
        safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_identifier))
        signal_file_per_model = self.signals_dir / f"rl_signals_{self.symbol}_{self.timeframe}_{safe_model_id}.log"
        
        # Добавляем timestamp
        signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Сохраняем в JSON формате
        payload = json.dumps(signal_data, ensure_ascii=False)
        # 1) Пишем в отдельный файл на модель (для корректной статистики и удобства)
        with open(signal_file_per_model, 'a', encoding='utf-8') as f:
            f.write(payload + '\n')

    def _save_latest_state(self, signal_data: Dict):
        """Сохраняет последний сигнал модели в отдельный JSON-файл.

        Каждый live‑агент перезаписывает только один файл на модель.
        Эти файлы затем читаются агрегатором ансамбля.
        """

        try:
            model_info = signal_data.get('model_info', {})
            model_identifier = model_info.get('model_identifier') or model_info.get('model_filename') or 'unknown_model'

            symbol = signal_data.get('symbol', 'UNKNOWN')
            timeframe = signal_data.get('timeframe', 'NA')

            state_filename = f"{symbol}_{timeframe}_{model_identifier}.json"
            state_path = self.live_state_dir / state_filename

            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, ensure_ascii=False)
        except Exception as e:
            # Ошибки состояния не должны ломать live‑агента
            print(f"⚠️ Could not save latest state: {e}")
    
    def predict_once(self) -> Dict:
        """Делает одно предсказание"""
        try:
            # Получаем данные
            df = self._fetch_latest_data()
            current_price = float(df['close'].iloc[-1])
            current_time = df.index[-1]

            # Для расчета тайминга свечи используем только текущий момент и таймфрейм
            now = datetime.now()
            candle_duration = self._timeframe_to_timedelta()

            # Рассчитываем ATR для SL/TP
            from ta.volatility import AverageTrueRange
            atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            atr = float(atr_indicator.average_true_range().iloc[-1])

            # === Приблизительное время с открытия текущей свечи и до её закрытия ===
            # Длительность бара в секундах
            duration_sec = int(candle_duration.total_seconds()) or 1

            # Считаем, сколько секунд прошло от последней "границы" сетки таймфрейма:
            #   elapsed_in_candle = now % duration
            now_ts = int(now.timestamp())
            elapsed_in_candle_sec = now_ts % duration_sec

            time_since_open = timedelta(seconds=elapsed_in_candle_sec)
            time_to_close = candle_duration - time_since_open
            if time_to_close < timedelta(0):
                time_to_close = timedelta(0)

            # Для JSON-логов храним секундные значения, а для консоли – формат HH:MM.
            time_since_open_sec = int(time_since_open.total_seconds())
            time_to_close_sec = int(time_to_close.total_seconds())

            # Подготавливаем observation через CryptoTradingEnv (как при обучении)
            obs = self._prepare_observation(df)
            


            # Получаем действие от агента (obs уже нормализован в _prepare_observation)
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Получаем вероятности действий из policy (используем ненормализованный obs)
            action_probs = self._get_action_probabilities(obs)
            
            signal = self._action_to_signal(int(action))
            
            # Рассчитываем SL/TP
            position_info = self._calculate_position_size(current_price, atr, signal)
            
            # Информация о модели (используем один раз и для логов, и для результата)
            model_info = self._get_model_info()

            # Формируем результат
            result = {
                'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol.replace('/', ''),
                'timeframe': self.timeframe,
                'signal': signal,
                'price': current_price,
                'atr': atr,
                'stop_loss': position_info['stop_loss'],
                'take_profit': position_info['take_profit'],
                'sl_multiplier': position_info['sl_multiplier'],
                'tp_multiplier': position_info['tp_multiplier'],
                'model_info': model_info,
                'agent_action': int(action),
                'action_probabilities': action_probs,
                'time_since_open_sec': time_since_open_sec,
                'time_to_close_sec': time_to_close_sec,
            }
            
            # Выводим в консоль с вероятностями (как XGBoost)
            # Используем такое же умное форматирование цены, как в Telegram
            status = "🟢 BUY" if signal == 'BUY' else "🔴 SELL" if signal == 'SELL' else "⚪ HOLD"
            price_str = format_price(current_price, self.symbol)
            proba_text = f"H:{action_probs['HOLD']:.1%} L:{action_probs['BUY']:.1%} S:{action_probs['SELL']:.1%}"
            model_folder = model_info.get('model_identifier', model_info.get('model_filename', 'N/A'))

            def _fmt_delta_short(td: timedelta) -> str:
                total_seconds = int(td.total_seconds())
                if total_seconds < 0:
                    total_seconds = 0
                minutes, _ = divmod(total_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                if hours > 99:
                    hours = 99
                return f"{hours:02d}:{minutes:02d}"

            print(
                f"{datetime.now().strftime('%H:%M:%S')} | {self.symbol} | {status} | {price_str} | "
                f"{proba_text} | {model_folder} | since_open={_fmt_delta_short(time_since_open)} | "
                f"to_close={_fmt_delta_short(time_to_close)}"
            )
            
            # Звуковой сигнал для торговых сигналов
            # Режимы:
            #   - no_sound=True: звук полностью отключён
            #   - signal_change_sound=True: бип только при смене сигнала
            #   - иначе: бип на каждый BUY/SELL
            if not self.no_sound and signal in ['BUY', 'SELL']:
                should_beep = False

                if self.signal_change_sound:
                    # Бип только если текущий сигнал отличается от предыдущего
                    if self.last_signal is None or signal != self.last_signal:
                        should_beep = True
                else:
                    # Обычный режим: всегда бипаем на BUY/SELL
                    should_beep = True

                if should_beep:
                    try:
                        # RL Agent: 1500 Hz, 300ms (выше и короче чем XGBoost 1000 Hz, 500ms)
                        winsound.Beep(1500, 300)
                    except Exception:
                        pass
            
            # Сохраняем в файл
            self._save_signal_to_file(result)

            # Обновляем файл с последним сигналом модели
            self._save_latest_state(result)

            # Обновляем last_signal / last_signal_time для логики звука
            # и возможного использования в будущем, независимо от Telegram.
            self.last_signal = signal
            self.last_signal_time = current_time

            # Отправляем в Telegram если нужно (без подробных логов в консоль)
            if self.telegram_enabled and self._should_send_signal(signal, current_time):
                try:
                    success = self._send_telegram_signal(result)

                    if not success:
                        # В случае неуспеха молчим, чтобы не спамить live‑консоль.
                        pass

                except Exception as e:
                    # Ошибку Telegram всё же выводим, это важная информация.
                    print(f"❌ Telegram error: {e}")
                    import traceback
                    traceback.print_exc()
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise
    
    def _send_telegram_signal(self, signal_data: Dict) -> bool:
        """Отправляет сигнал в Telegram"""
        try:
            # Импортируем обе функции для fallback
            from telegram_sender import send_trading_signal, send_telegram_message
            
            # Форматируем для telegram_sender.send_trading_signal()
            # Используем формат, который ожидает format_trading_signal()
            telegram_data = {
                'symbol': signal_data['symbol'],
                'timeframe': signal_data['timeframe'],
                'signal': signal_data['signal'],
                'timestamp': signal_data['time'],
                'price': signal_data['price'],
                'stop_loss': signal_data['stop_loss'],
                'take_profit': signal_data['take_profit'],
                'confidence': max(signal_data['action_probabilities'].values()),
                'model_identifier': signal_data['model_info'].get('model_identifier', signal_data['model_info'].get('model_name')),
                'model_filename': signal_data['model_info'].get('model_filename', signal_data['model_info'].get('model_name')),
                'details': {
                    'HOLD': signal_data['action_probabilities'].get('HOLD', 0),
                    'LONG': signal_data['action_probabilities'].get('BUY', 0),
                    'SHORT': signal_data['action_probabilities'].get('SELL', 0),
                },
            }
            
            # Пробуем отправить через send_trading_signal
            success = send_trading_signal(telegram_data)
            
            if not success:
                # Fallback: отправляем простое сообщение
                print("⚠️  send_trading_signal returned False, trying fallback...")
                
                signal_icon = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
                
                simple_message = f"""
{signal_icon} <b>RL AGENT SIGNAL</b>

<b>Symbol:</b> {signal_data['symbol']} ({signal_data['timeframe']})
<b>Signal:</b> {signal_data['signal']}
<b>Price:</b> ${signal_data['price']:,.2f}
<b>Time:</b> {signal_data['time']}

📊 <b>Levels:</b>
Stop Loss: ${signal_data['stop_loss']:,.2f}
Take Profit: ${signal_data['take_profit']:,.2f}

🎯 <b>Model:</b> {signal_data['model_info']['algorithm']}
Model ID: {signal_data['model_info'].get('model_identifier', signal_data['model_info'].get('model_name'))}
File: {signal_data['model_info'].get('model_filename', '')}
Return: {signal_data['model_info']['performance'].get('final_return', 0):+.2%}
                """.strip()
                
                success = send_telegram_message(simple_message)
                
            return success
            
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Запускает live trading"""
        print(f"🚀 Starting RL Agent...")
        print(f"⏱️  Check interval: {self._format_interval()}")
        print(f"Press Ctrl+C to stop\n")
        
        if not self.continuous:
            # Одноразовое предсказание
            self.predict_once()
            return
        
        # Непрерывный мониторинг
        while True:
            try:
                self.predict_once()
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                print(f"\n\n🛑 Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"🔄 Retrying in {self._format_interval()}...")
                time.sleep(self.interval)


def main():
    parser = argparse.ArgumentParser(
        description='🤖 RL Agent Live Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Single prediction (A2C 4h model):
   python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800

2. Continuous monitoring (PPO 1h model):
   python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_1h_PPO_20251104_115146 --continuous

3. With Telegram notifications:
   python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800 --continuous --telegram

4. Custom symbol and timeframe:
   python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800 --symbol ETHUSDT --timeframe 4h --continuous

5. Custom check interval (10 minutes):
   python rl_system/run_live_agent.py --model-path rl_system/models/BTCUSDT_4h_A2C_20251104_113800 --continuous --interval 600
        """
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory (e.g., rl_system/models/BTCUSDT_4h_A2C_20251104_113800)')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Trading symbol (default: from model config)')
    parser.add_argument('--timeframe', type=str, default=None,
                       help='Timeframe (default: from model config)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous monitoring (default: single prediction)')
    parser.add_argument('--telegram', action='store_true',
                       help='Enable Telegram notifications')
    parser.add_argument('--interval', type=int, default=None,
                       help='Check interval in seconds (default: based on timeframe)')
    parser.add_argument('--no-sound', action='store_true',
                       help='Disable audible beep on signals')
    parser.add_argument('--signal-change-sound', action='store_true',
                       help='Beep only when signal changes (HOLD->BUY, BUY->SELL, etc.)')
    
    args = parser.parse_args()
    
    # Создаем и запускаем трейдера
    trader = RLLiveTrader(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        telegram_enabled=args.telegram,
        continuous=args.continuous,
        interval=args.interval,
        no_sound=args.no_sound,
        signal_change_sound=args.signal_change_sound
    )
    
    trader.run()


if __name__ == '__main__':
    main()
