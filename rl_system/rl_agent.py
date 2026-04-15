"""
RL Agent Wrapper
================

Удобный интерфейс для работы с обученным агентом.
"""

from pathlib import Path
import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from trading_env import MarketTradingEnv, Actions, Positions


class RLAgent:
    """
    Wrapper для обученного RL агента.
    
    Упрощает использование модели для:
    - Получения торговых сигналов
    - Backtesting
    - Live trading
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Путь к директории с моделью
        """
        self.model_path = Path(model_path)
        self.model = None
        self.config = {}
        self.vec_normalize = None
        self.engineer = FeatureEngineer()
        
        self._load_model()
    
    def _load_model(self):
        """Загружает модель и конфигурацию."""
        # Конфиг
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Модель (best или final)
        best_model = self.model_path / "best" / "best_model.zip"
        final_model = self.model_path / "final_model.zip"
        
        if best_model.exists():
            self.model = PPO.load(best_model)
            print(f"✅ Loaded best model from {best_model}")
        elif final_model.exists():
            self.model = PPO.load(final_model)
            print(f"✅ Loaded final model from {final_model}")
        else:
            raise FileNotFoundError(f"No model found in {self.model_path}")
        
        # VecNormalize - load path only, will wrap environment later
        vec_norm_path = self.model_path / "vec_normalize.pkl"
        if vec_norm_path.exists():
            self.vec_normalize_path = vec_norm_path
            self.vec_normalize = None  # Will be loaded when wrapping environment
            print(f"✅ Found VecNormalize at {vec_norm_path}")
        else:
            self.vec_normalize_path = None
            self.vec_normalize = None
    
    def predict_action(self, 
                      df: pd.DataFrame, 
                      current_position: Positions = Positions.FLAT,
                      deterministic: bool = True,
                      entry_price: Optional[float] = None,
                      balance: float = 10000.0) -> Tuple[Actions, float]:
        """
        Предсказывает действие на основе данных.
        
        Args:
            df: DataFrame с последними данными (минимум lookback_window баров)
            current_position: Текущая позиция
            deterministic: Использовать детерминированное действие
            entry_price: Цена входа (если в позиции)
            balance: Текущий баланс
        
        Returns:
            (action, confidence)
        """
        # Подготовка фичей
        df_features = self.engineer.calculate_features(df)
        df_features = self.engineer.normalize_features(df_features, method='zscore')
        
        # Создаем временное окружение для получения observation
        lookback = self.config.get('lookback', 60)
        initial_balance = self.config.get('initial_balance', 10000.0)
        
        if len(df_features) < lookback:
            raise ValueError(f"Need at least {lookback} bars, got {len(df_features)}")
        
        # Берем последние lookback баров
        window_df = df_features.iloc[-lookback:].copy()
        current_price = df.iloc[-1]['close']
        
        # Формируем observation
        feature_cols = [col for col in window_df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        features = window_df[feature_cols].values
        
        # Вычисляем реалистичную position info (КАК ПРИ ОБУЧЕНИИ!)
        equity = balance
        unrealized_pnl = 0.0
        
        if current_position != Positions.FLAT and entry_price is not None:
            # Приблизительно считаем position_quantity
            position_quantity = balance / entry_price
            
            if current_position == Positions.LONG:
                unrealized_pnl = (current_price - entry_price) * position_quantity
            elif current_position == Positions.SHORT:
                unrealized_pnl = (entry_price - current_price) * position_quantity
            
            equity = balance + unrealized_pnl
        
        # Формируем position_info ТОЧНО КАК В trading_env.py
        position_info = np.array([
            current_position.value,                    # position (-1, 0, 1)
            equity / initial_balance,                  # normalized equity
            0.0,                                       # total_profit (неизвестно в inference)
            unrealized_pnl / initial_balance,          # unrealized pnl normalized
            0.0                                        # max_drawdown (неизвестно в inference)
        ])
        
        # Повторяем position_info для каждого временного шага
        position_features = np.tile(position_info, (lookback, 1))
        
        observation = np.concatenate([features, position_features], axis=1).astype(np.float32)
        observation = observation.reshape(1, lookback, -1)
        
        # Применяем VecNormalize если есть (загружаем лениво при первом вызове)
        if self.vec_normalize_path and self.vec_normalize is None:
            # Create dummy env to load VecNormalize stats
            import sys
            from pathlib import Path
            # Add parent directory to path for imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from rl_system.trading_env import MarketTradingEnv
            
            # Only pass relevant arguments to MarketTradingEnv
            env_kwargs = {
                'initial_balance': self.config.get('initial_balance', 10000.0),
                'commission': self.config.get('commission', 0.001),
                'slippage': self.config.get('slippage', 0.0005),
                'lookback_window': self.config.get('lookback', 60),
                'position_size': self.config.get('position_size', 1.0),
                'enable_short': self.config.get('enable_short', True)
            }
            dummy_env = MarketTradingEnv(df_features.copy(), **env_kwargs)
            dummy_vec_env = DummyVecEnv([lambda: dummy_env])
            self.vec_normalize = VecNormalize.load(self.vec_normalize_path, dummy_vec_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print(f"✅ Loaded VecNormalize for inference")
        
        if self.vec_normalize:
            observation = self.vec_normalize.normalize_obs(observation)
        
        # Предсказание
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Конвертируем в Actions enum
        action_value = action[0] if isinstance(action, np.ndarray) else action
        
        # Confidence (упрощенная версия - можно улучшить через policy.predict_values)
        confidence = 0.8 if deterministic else 0.6
        
        return Actions(action_value), confidence
    
    def generate_signal(self, 
                       symbol: str, 
                       timeframe: str = '30m',
                       current_position: str = 'FLAT',
                       entry_price: Optional[float] = None,
                       balance: float = 10000.0) -> dict:
        """
        Генерирует торговый сигнал для символа.
        
        Args:
            symbol: Торговая пара (e.g., 'BTCUSDT')
            timeframe: Таймфрейм
            current_position: Текущая позиция ('FLAT', 'LONG', 'SHORT')
            entry_price: Цена входа (если в позиции)
            balance: Текущий баланс (для расчета unrealized PnL)
        
        Returns:
            Dict с сигналом
        """
        # Загружаем свежие данные
        loader = DataLoader()
        lookback = self.config.get('lookback', 60)
        df = loader.load_data(symbol, timeframe, days=30, use_cache=False)
        
        # Берем последние данные
        df = df.tail(lookback + 50)  # +50 для расчета индикаторов
        
        # Конвертируем текущую позицию
        pos_map = {'FLAT': Positions.FLAT, 'LONG': Positions.LONG, 'SHORT': Positions.SHORT}
        current_pos = pos_map.get(current_position.upper(), Positions.FLAT)
        
        # Получаем предсказание (передаем entry_price и balance)
        action, confidence = self.predict_action(
            df, current_pos, deterministic=True,
            entry_price=entry_price, balance=balance
        )
        
        # Формируем сигнал
        current_price = df.iloc[-1]['close']
        
        # Вычисляем unrealized PnL если в позиции
        unrealized_pnl = None
        if current_pos != Positions.FLAT and entry_price is not None:
            if current_pos == Positions.LONG:
                unrealized_pnl = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                unrealized_pnl = ((entry_price - current_price) / entry_price) * 100
        
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': df.iloc[-1]['timestamp'],
            'price': current_price,
            'action': action.name,  # 'BUY', 'SELL', 'HOLD'
            'current_position': current_position,
            'entry_price': entry_price,
            'unrealized_pnl': unrealized_pnl,
            'confidence': confidence,
            'model': self.model_path.name,
            'recommended_action': self._get_recommended_action(action, current_pos)
        }
        
        return signal
    
    def _get_recommended_action(self, action: Actions, current_position: Positions) -> str:
        """Интерпретирует действие в рекомендацию."""
        if action == Actions.BUY:
            if current_position == Positions.FLAT:
                return "OPEN_LONG"
            elif current_position == Positions.SHORT:
                return "CLOSE_SHORT"
            else:
                return "HOLD_LONG"
        
        elif action == Actions.SELL:
            if current_position == Positions.FLAT:
                return "OPEN_SHORT"
            elif current_position == Positions.LONG:
                return "CLOSE_LONG"
            else:
                return "HOLD_SHORT"
        
        else:  # HOLD
            return "HOLD"
    
    @classmethod
    def load(cls, model_path: str) -> 'RLAgent':
        """Загружает агента из директории."""
        return cls(model_path)
    
    def get_config(self) -> dict:
        """Возвращает конфигурацию модели."""
        return self.config.copy()
    
    def __repr__(self):
        return f"RLAgent(model={self.model_path.name}, symbol={self.config.get('symbol', 'N/A')})"


# Пример использования
if __name__ == "__main__":
    # Загружаем агента
    agent = RLAgent.load("rl_system/models/BTCUSDT_30m_20241102_120000")
    
    print(f"Agent: {agent}")
    print(f"Config: {agent.get_config()}")
    
    # Генерируем сигнал
    signal = agent.generate_signal('BTCUSDT', timeframe='30m', current_position='FLAT')
    
    print(f"\n📊 Signal:")
    for key, value in signal.items():
        print(f"  {key}: {value}")
