"""
Configuration Manager для RL Trading System
============================================

Управление пресетами и конфигурациями.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Управление конфигурациями для RL системы."""
    
    # Настройки для расчета минимального количества сделок
    # Используется универсальная формула: max(STATISTICAL_MINIMUM, months × TRADES_PER_MONTH)
    # Не зависит от таймфрейма - статистическая значимость одинакова при одинаковом N
    MIN_TRADES_SETTINGS = {
        'STATISTICAL_MINIMUM': 30,  # Абсолютный минимум для статистической значимости (±18% погрешность)
        'TRADES_PER_MONTH': 10,     # Ожидаемая активность модели (~2.5 сделки в неделю)
    }
    
    PRESETS = {
        'conservative': {
            'name': 'Conservative',
            'description': 'Минимальный риск, высокое качество сигналов',
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'lookback_window': 60,
            'position_size': 0.80,  # 80% capital per trade
            'enable_short': False,
            'learning_rate': 1e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'clip_range': 0.1,
        },
        
        'balanced': {
            'name': 'Balanced',
            'description': 'Сбалансированный риск/доходность',
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'lookback_window': 60,
            'position_size': 0.95,  # 95% capital
            'enable_short': True,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'clip_range': 0.2,
        },
        
        'aggressive': {
            'name': 'Aggressive',
            'description': 'Максимальная доходность, высокий риск',
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'lookback_window': 40,  # Shorter lookback
            'position_size': 1.0,  # 100% capital
            'enable_short': True,
            'learning_rate': 5e-4,
            'n_steps': 2048,
            'batch_size': 128,
            'n_epochs': 20,
            'gamma': 0.95,  # Less emphasis on future
            'clip_range': 0.3,
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        Получает preset конфигурацию.
        
        Args:
            preset_name: 'conservative', 'balanced', или 'aggressive'
        
        Returns:
            Dict с параметрами
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def list_presets(cls):
        """Выводит список доступных пресетов."""
        print("\n📋 Available Presets:\n")
        for name, config in cls.PRESETS.items():
            print(f"  • {name.upper()}")
            print(f"    {config['description']}")
            print(f"    Position Size: {config['position_size']*100:.0f}%")
            print(f"    Short Enabled: {config['enable_short']}")
            print(f"    Learning Rate: {config['learning_rate']}")
            print()
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], filepath: str):
        """Сохраняет конфигурацию в JSON."""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"💾 Config saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict[str, Any]:
        """Загружает конфигурацию из JSON."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"📦 Config loaded from {filepath}")
        return config
    
    @classmethod
    def create_custom_config(cls,
                            base_preset: str = 'balanced',
                            **overrides) -> Dict[str, Any]:
        """
        Создает кастомную конфигурацию на основе пресета.
        
        Args:
            base_preset: Базовый пресет
            **overrides: Параметры для переопределения
        
        Returns:
            Dict с конфигурацией
        """
        config = cls.get_preset(base_preset)
        config.update(overrides)
        return config


# Пример использования
if __name__ == "__main__":
    # Список пресетов
    ConfigManager.list_presets()
    
    # Получить пресет
    conservative = ConfigManager.get_preset('conservative')
    print(f"Conservative config: {conservative}")
    
    # Создать кастомную конфигурацию
    custom = ConfigManager.create_custom_config(
        base_preset='balanced',
        position_size=0.5,
        learning_rate=1e-3
    )
    print(f"\nCustom config: {custom}")
    
    # Сохранить
    ConfigManager.save_config(custom, 'rl_system/configs/my_config.json')
    
    # Загрузить
    loaded = ConfigManager.load_config('rl_system/configs/my_config.json')
    print(f"\nLoaded config: {loaded}")
