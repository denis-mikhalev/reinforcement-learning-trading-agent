"""
RL Signal Generator - Генератор сигналов для ручной торговли
=============================================================

Простой скрипт для получения торговых сигналов от RL агента.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Добавляем путь к rl_system
sys.path.insert(0, str(Path(__file__).parent))

from rl_agent import RLAgent


class RLSignalGenerator:
    """Генератор сигналов для ручной торговли."""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Путь к обученной модели
        """
        self.agent = RLAgent(model_path)
        self.current_position = 'FLAT'
        self.entry_price = None
        self.entry_time = None
        self.balance = 10000.0
    
    def check_signal(self, symbol: str, timeframe: str = '30m') -> dict:
        """
        Проверяет сигнал для символа.
        
        Args:
            symbol: Торговая пара (e.g., 'BTCUSDT')
            timeframe: Таймфрейм
        
        Returns:
            Dict с сигналом и интерпретацией
        """
        # Получаем сигнал от агента
        signal = self.agent.generate_signal(
            symbol=symbol,
            timeframe=timeframe,
            current_position=self.current_position,
            entry_price=self.entry_price,
            balance=self.balance
        )
        
        # Добавляем информацию о текущей позиции
        result = {
            **signal,
            'our_position': self.current_position,
            'our_entry_price': self.entry_price,
            'our_entry_time': self.entry_time,
            'interpretation': self._interpret_signal(signal)
        }
        
        return result
    
    def _interpret_signal(self, signal: dict) -> str:
        """Интерпретирует сигнал для человека."""
        action = signal['recommended_action']
        
        if action == 'OPEN_LONG':
            return f"🟢 ОТКРЫТЬ LONG по {signal['price']:.2f}"
        
        elif action == 'OPEN_SHORT':
            return f"🔴 ОТКРЫТЬ SHORT по {signal['price']:.2f}"
        
        elif action == 'CLOSE_LONG':
            pnl = signal.get('unrealized_pnl', 0)
            emoji = '🟢' if pnl > 0 else '🔴'
            return f"{emoji} ЗАКРЫТЬ LONG по {signal['price']:.2f} (P&L: {pnl:+.2f}%)"
        
        elif action == 'CLOSE_SHORT':
            pnl = signal.get('unrealized_pnl', 0)
            emoji = '🟢' if pnl > 0 else '🔴'
            return f"{emoji} ЗАКРЫТЬ SHORT по {signal['price']:.2f} (P&L: {pnl:+.2f}%)"
        
        elif action == 'HOLD':
            if self.current_position == 'FLAT':
                return "⏸️ НЕТ СИГНАЛА (ждем подходящего момента)"
            else:
                pnl = signal.get('unrealized_pnl', 0)
                return f"⏸️ ДЕРЖАТЬ {self.current_position} (Unrealized P&L: {pnl:+.2f}%)"
        
        return "❓ Неизвестное действие"
    
    def update_position(self, position: str, entry_price: Optional[float] = None):
        """
        Обновляет текущую позицию после действия на бирже.
        
        Args:
            position: 'FLAT', 'LONG', 'SHORT'
            entry_price: Цена входа (если открыли позицию)
        """
        self.current_position = position.upper()
        
        if position.upper() != 'FLAT' and entry_price is not None:
            self.entry_price = entry_price
            self.entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            self.entry_price = None
            self.entry_time = None
        
        print(f"✅ Позиция обновлена: {self.current_position}")
        if self.entry_price:
            print(f"📊 Цена входа: {self.entry_price:.2f}")
            print(f"⏰ Время входа: {self.entry_time}")
    
    def print_signal(self, signal: dict):
        """Красиво выводит сигнал."""
        print("\n" + "="*70)
        print(f"🎯 RL AGENT SIGNAL")
        print("="*70)
        
        print(f"\n📊 РЫНОЧНАЯ ИНФОРМАЦИЯ:")
        print(f"   Символ: {signal['symbol']}")
        print(f"   Таймфрейм: {signal['timeframe']}")
        print(f"   Текущая цена: {signal['price']:.2f}")
        print(f"   Время: {signal['timestamp']}")
        
        print(f"\n💼 ТЕКУЩАЯ ПОЗИЦИЯ:")
        print(f"   Позиция: {signal['our_position']}")
        if signal['our_entry_price']:
            print(f"   Цена входа: {signal['our_entry_price']:.2f}")
            print(f"   Время входа: {signal['our_entry_time']}")
            if signal.get('unrealized_pnl') is not None:
                emoji = '📈' if signal['unrealized_pnl'] > 0 else '📉'
                print(f"   {emoji} Unrealized P&L: {signal['unrealized_pnl']:+.2f}%")
        
        print(f"\n🤖 РЕКОМЕНДАЦИЯ АГЕНТА:")
        print(f"   {signal['interpretation']}")
        print(f"   Уверенность: {signal['confidence']:.0%}")
        print(f"   Модель: {signal['model']}")
        
        print("\n" + "="*70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='RL Agent Signal Generator - Генератор торговых сигналов'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к обученной модели (e.g., rl_system/models/BTCUSDT_30m_20251102_120000)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Торговая пара (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='30m',
        help='Таймфрейм (default: 30m)'
    )
    
    parser.add_argument(
        '--position',
        type=str,
        default='FLAT',
        choices=['FLAT', 'LONG', 'SHORT'],
        help='Текущая позиция (default: FLAT)'
    )
    
    parser.add_argument(
        '--entry-price',
        type=float,
        default=None,
        help='Цена входа (если в позиции)'
    )
    
    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Текущий баланс (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Создаем генератор
    print("🚀 Инициализация RL Signal Generator...")
    generator = RLSignalGenerator(args.model)
    
    # Устанавливаем текущую позицию
    if args.position != 'FLAT':
        if args.entry_price is None:
            print("❌ Ошибка: Для позиции LONG/SHORT нужно указать --entry-price")
            sys.exit(1)
        
        generator.update_position(args.position, args.entry_price)
    
    generator.balance = args.balance
    
    # Проверяем сигнал
    print(f"\n🔍 Проверка сигнала для {args.symbol} {args.timeframe}...")
    
    try:
        signal = generator.check_signal(args.symbol, args.timeframe)
        generator.print_signal(signal)
        
        print("\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        
        if signal['recommended_action'] in ['OPEN_LONG', 'OPEN_SHORT']:
            print("   1. Откройте позицию на бирже")
            print("   2. Запустите скрипт снова с параметрами:")
            print(f"      python rl_signal_generator.py \\")
            print(f"          --model {args.model} \\")
            print(f"          --symbol {args.symbol} \\")
            print(f"          --position LONG \\")
            print(f"          --entry-price <цена_входа>")
        
        elif signal['recommended_action'] in ['CLOSE_LONG', 'CLOSE_SHORT']:
            print("   1. Закройте позицию на бирже")
            print("   2. Запустите скрипт снова с --position FLAT")
        
        elif signal['recommended_action'] == 'HOLD':
            if generator.current_position == 'FLAT':
                print("   Пока нет сигнала для входа")
                print("   Проверьте снова через 30 минут")
            else:
                print("   Продолжайте держать позицию")
                print("   Проверьте снова через 30 минут")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка при получении сигнала: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
