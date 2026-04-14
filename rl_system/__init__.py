"""
Reinforcement Learning Trading System
======================================

Автономная система на базе PPO для криптотрейдинга.
"""

__version__ = "1.0.0"
__author__ = "Denis Mikhalev"

from .trading_env import CryptoTradingEnv
from .rl_agent import RLAgent
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer

__all__ = [
    "CryptoTradingEnv",
    "RLAgent",
    "DataLoader",
    "FeatureEngineer",
]
