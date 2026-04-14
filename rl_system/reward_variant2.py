"""
ВАРИАНТ 2: Увеличенный reward за HOLD profitable позиций

Изменения:
- unrealized_pnl * 0.0002 → unrealized_pnl * 0.02 (×100)
- Делает holding прибыльных позиций гораздо более привлекательным
- Открытие позиции: небольшой штраф -7.5 (как было изначально)
"""

# Вставить в trading_env.py в метод _execute_action:

def _execute_action(self, action: int, price: float) -> float:
    """
    Выполняет торговое действие и возвращает награду.
    
    Reward function v5 - VARIANT 2: High HOLD reward
    - Realized PnL от закрытия позиции (основная награда)
    - Малый штраф за открытие позиции (0.5x от реальных издержек)
    - БОЛЬШАЯ награда за holding profitable position (×100)
    
    Returns:
        reward (absolute dollar value)
    """
    reward = 0.0
    
    # SELL Action
    if action == Actions.SELL:
        if self.position == Positions.LONG:
            # Закрываем long - получаем realized PnL
            reward = self._close_position(price)
        elif self.position == Positions.FLAT and self.enable_short:
            # Открываем short - уменьшенный штраф (50% от полных издержек)
            self._open_short(price)
            reward = -self.initial_balance * (self.commission + self.slippage) * 0.5
    
    # BUY Action
    elif action == Actions.BUY:
        if self.position == Positions.SHORT:
            # Закрываем short - получаем realized PnL
            reward = self._close_position(price)
        elif self.position == Positions.FLAT:
            # Открываем long - уменьшенный штраф (50% от полных издержек)
            self._open_long(price)
            reward = -self.initial_balance * (self.commission + self.slippage) * 0.5
    
    # HOLD - БОЛЬШАЯ reward за удержание прибыльной позиции
    elif action == Actions.HOLD:
        if self.position == Positions.LONG:
            unrealized_pnl = (price - self.entry_price) * self.position_quantity
            if unrealized_pnl > 0:
                reward = unrealized_pnl * 0.02  # VARIANT 2: 2% от текущей прибыли (×100!)
        elif self.position == Positions.SHORT:
            unrealized_pnl = (self.entry_price - price) * self.position_quantity
            if unrealized_pnl > 0:
                reward = unrealized_pnl * 0.02  # VARIANT 2: 2% от текущей прибыли (×100!)
        # FLAT позиция: нет награды, нет штрафа (агент сам решает когда торговать)
    
    return reward
