"""
ВАРИАНТ 3: Награда только за realized PnL (закрытие позиций)

Изменения:
- HOLD в любой позиции: reward = 0 (нет промежуточных наград)
- Открытие позиции: reward = 0 (нейтрально)
- Закрытие позиции: reward = realized PnL (вся прибыль/убыток)
- Простая и понятная логика: агент получает reward только когда реализует результат
"""

# Вставить в trading_env.py в метод _execute_action:

def _execute_action(self, action: int, price: float) -> float:
    """
    Выполняет торговое действие и возвращает награду.
    
    Reward function v5 - VARIANT 3: Only realized PnL
    - Награда ТОЛЬКО за закрытие позиции (realized PnL)
    - Открытие позиции: reward = 0 (нейтрально)
    - HOLD: reward = 0 (никаких промежуточных наград)
    - Простая логика: reward = финальный результат сделки
    
    Returns:
        reward (absolute dollar value)
    """
    reward = 0.0
    
    # SELL Action
    if action == Actions.SELL:
        if self.position == Positions.LONG:
            # Закрываем long - получаем realized PnL (ВСЯ награда здесь!)
            reward = self._close_position(price)
        elif self.position == Positions.FLAT and self.enable_short:
            # Открываем short - нейтрально
            self._open_short(price)
            reward = 0.0  # VARIANT 3: No penalty, no reward
    
    # BUY Action
    elif action == Actions.BUY:
        if self.position == Positions.SHORT:
            # Закрываем short - получаем realized PnL (ВСЯ награда здесь!)
            reward = self._close_position(price)
        elif self.position == Positions.FLAT:
            # Открываем long - нейтрально
            self._open_long(price)
            reward = 0.0  # VARIANT 3: No penalty, no reward
    
    # HOLD - НИКАКИХ наград (агент должен сам решить когда закрыть)
    elif action == Actions.HOLD:
        reward = 0.0  # VARIANT 3: No intermediate rewards!
        # Агент не получает подсказок о прибыльности текущей позиции
        # Он должен сам научиться оптимальному времени закрытия
    
    return reward
