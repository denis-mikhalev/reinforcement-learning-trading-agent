"""
Trading Environment для Reinforcement Learning
===============================================

Gymnasium-совместимое окружение для обучения агента.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from enum import IntEnum


class Actions(IntEnum):
    """Возможные действия агента."""
    SELL = 0    # Short или выход из long
    HOLD = 1    # Ничего не делать
    BUY = 2     # Long или выход из short


class Positions(IntEnum):
    """Текущая позиция."""
    SHORT = -1
    FLAT = 0
    LONG = 1


class MarketTradingEnv(gym.Env):
    """
    Окружение для крипто-трейдинга с поддержкой:
    - Long/Short позиций
    - Комиссий и проскальзывания
    - Реалистичного управления позициями
    - Детальной статистики
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 lookback_window: int = 60,
                 position_size: float = 1.0,
                 enable_short: bool = True,
                 max_episode_steps: Optional[int] = None,
                 stop_loss_pct: float = 0.0,
                 max_holding_bars: int = 0,
                 hold_flat_penalty: float = 0.00005,
                 min_trades_target: int = 0,
                 too_few_trades_penalty: float = 0.0):
        """
        Args:
            df: DataFrame с фичами (уже подготовленными feature_engineering)
            initial_balance: Начальный баланс в USDT
            commission: Комиссия за сделку (0.001 = 0.1%)
            slippage: Проскальзывание (0.0005 = 0.05%)
            lookback_window: Количество баров для observation
            position_size: Размер позиции (1.0 = 100% баланса)
            enable_short: Разрешить шорт-позиции
            max_episode_steps: Максимальная длина эпизода
            stop_loss_pct: Stop-loss в % от баланса (0.01 = 1%, 0 = disabled)
            max_holding_bars: Максимальное кол-во баров удержания позиции (0 = disabled)
                              Например: 48 баров на 4h = 8 дней, 72 бара = 12 дней
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.lookback_window = lookback_window
        self.position_size = position_size
        self.enable_short = enable_short
        self.max_episode_steps = max_episode_steps or len(df) - lookback_window - 1
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars

        # Reward shaping knobs (defaults preserve current behavior)
        self.hold_flat_penalty = float(abs(hold_flat_penalty))
        self.min_trades_target = int(min_trades_target) if min_trades_target is not None else 0
        self.too_few_trades_penalty = float(max(0.0, too_few_trades_penalty))
        
        # Определяем пространства
        self._setup_spaces()
        
        # Инициализация состояния
        self.reset()
    
    def _calculate_global_metrics(self):
        """Рассчитывает глобальные метрики для всей истории (не зависят от lookback)."""
        prices = self.df['close'].values
        
        # ATH/ATL для всей истории
        self.global_ath = np.max(prices)
        self.global_atl = np.min(prices)
        self.global_ath_idx = np.argmax(prices)
        self.global_atl_idx = np.argmin(prices)
        
        # Рассчитываем percentile для каждого бара относительно ВСЕЙ истории
        # Для каждой цены находим её позицию в отсортированном массиве всех цен
        self.price_percentiles = np.zeros(len(prices))
        sorted_prices = np.sort(prices)
        for i, price in enumerate(prices):
            self.price_percentiles[i] = np.searchsorted(sorted_prices, price) / len(sorted_prices)
        
        # Находим major support/resistance levels используя volume profile
        # Берем цены с высоким volume как значимые уровни
        if 'volume' in self.df.columns:
            volume = self.df['volume'].values
            
            # Создаем price bins и суммируем volume в них
            n_bins = 50
            price_range = self.global_ath - self.global_atl
            bin_size = price_range / n_bins
            
            # Volume Profile: сколько volume торговалось на каждом уровне
            volume_profile = np.zeros(n_bins)
            for price, vol in zip(prices, volume):
                bin_idx = int((price - self.global_atl) / bin_size)
                if 0 <= bin_idx < n_bins:
                    volume_profile[bin_idx] += vol
            
            # Находим топ-5 уровней с максимальным volume (POC - Point of Control)
            top_volume_bins = np.argsort(volume_profile)[-5:]
            self.major_levels = sorted([self.global_atl + (bin_idx + 0.5) * bin_size 
                                       for bin_idx in top_volume_bins])
        else:
            # Если нет volume, используем просто price quartiles
            self.major_levels = [
                np.percentile(prices, 20),
                np.percentile(prices, 40),
                np.percentile(prices, 60),
                np.percentile(prices, 80)
            ]
    
    def _setup_spaces(self):
        """Настройка observation и action spaces."""
        # Определяем количество фичей (все колонки кроме служебных)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        # +5 для position info (position, equity, pnl, unrealized_pnl, drawdown)
        # Multi-timeframe features уже включены в feature_cols!
        self.n_features = len(feature_cols) + 5
        
        # Observation space: [lookback_window, n_features]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.n_features),
            dtype=np.float32
        )
        
        # Action space: Discrete(3) для SELL/HOLD/BUY
        self.action_space = spaces.Discrete(3)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Сброс окружения."""
        super().reset(seed=seed)
        
        # Начинаем с lookback_window, чтобы было достаточно истории
        self.current_step = self.lookback_window
        
        # Финансовые переменные
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = Positions.FLAT
        self.entry_price = 0.0
        self.position_quantity = 0.0
        self.position_holding_bars = 0  # NEW: Tracking holding duration (for V4)
        
        # Статистика
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0  # Сумма всех прибыльных сделок
        self.gross_loss = 0.0    # Сумма всех убыточных сделок
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        
        # История для расчета наград
        self.equity_history = [self.initial_balance]
        self.trade_history = []

        # Детальный лог сделок для анализа бэктестов
        # Не используется в награде, только для оффлайн-аналитики
        self.trades_log = []

        # Reward instrumentation: разложение reward по компонентам (суммы за эпизод)
        self.reward_breakdown_sums = self._empty_reward_breakdown()
        self.reward_total_sum = 0.0
        self.reward_steps = 0
        self.last_reward_breakdown = self._empty_reward_breakdown()
        
        return self._get_observation(), self._get_info()

    def _empty_reward_breakdown(self) -> Dict[str, float]:
        return {
            'realized_close': 0.0,
            'open_penalty': 0.0,
            'hold_unrealized': 0.0,
            'hold_flat_penalty': 0.0,
            'too_few_trades_penalty': 0.0,
            'forced_stop_loss_close': 0.0,
            'forced_max_holding_close': 0.0,
            'forced_end_close': 0.0,
            'terminal_bonus': 0.0,
            'dd_bonus': 0.0,
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Выполнение действия.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Текущая цена
        current_price = self.df.iloc[self.current_step]['close']
        
        # Применяем проскальзывание
        if action == Actions.BUY:
            execution_price = current_price * (1 + self.slippage)
        elif action == Actions.SELL:
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price
        
        # Выполняем действие + получаем разложение reward
        reward, reward_breakdown = self._execute_action(action, execution_price)
        
        # Обновляем equity (с учетом unrealized PnL)
        self._update_equity(current_price)
        
        # STOP-LOSS: Проверяем убыток и принудительно закрываем если превышен
        # Используем worst-case цену (low для LONG, high для SHORT) чтобы не "проскочить" stop-loss
        if self.stop_loss_pct > 0 and self.position != Positions.FLAT:
            # Stop-loss триггерится если убыток > stop_loss_pct от НАЧАЛЬНОГО баланса
            stop_loss_threshold = -self.initial_balance * self.stop_loss_pct
            
            # Для LONG проверяем по LOW цене бара (худший случай)
            # Для SHORT проверяем по HIGH цене бара (худший случай)
            if self.position == Positions.LONG:
                worst_price = self.df.iloc[self.current_step]['low']
            else:  # SHORT
                worst_price = self.df.iloc[self.current_step]['high']
            
            unrealized_pnl_worst = self._calculate_unrealized_pnl(worst_price)
            
            if unrealized_pnl_worst < stop_loss_threshold:
                # Закрываем по stop-loss цене.
                # ВАЖНО: цена должна быть реалистичной (внутри бара) и не уходить в отрицательные значения,
                # иначе при очень маленьком position_quantity можно получить огромный и некорректный PnL.
                qty = float(self.position_quantity)
                if (not np.isfinite(qty)) or qty <= 1e-12:
                    stop_loss_price = float(worst_price)
                else:
                    stop_loss_amount_per_unit = (self.initial_balance * self.stop_loss_pct) / qty
                    if self.position == Positions.LONG:
                        # Для LONG stop-loss не может быть ниже low бара и выше entry
                        stop_loss_price = self.entry_price - stop_loss_amount_per_unit
                        stop_loss_price = max(float(worst_price), min(float(self.entry_price), float(stop_loss_price)))
                    else:  # SHORT
                        # Для SHORT stop-loss не может быть выше high бара и ниже entry
                        stop_loss_price = self.entry_price + stop_loss_amount_per_unit
                        stop_loss_price = min(float(worst_price), max(float(self.entry_price), float(stop_loss_price)))

                stop_loss_price = max(1e-12, float(stop_loss_price))

                # Принудительно закрываем позицию по stop-loss цене
                stop_loss_pnl = self._close_position(stop_loss_price)
                # Нормализуем reward как и в _execute_action
                stop_loss_reward = stop_loss_pnl / self.initial_balance
                reward += stop_loss_reward
                reward_breakdown['forced_stop_loss_close'] += stop_loss_reward
                self._update_equity(current_price)  # Обновляем equity после закрытия
        
        # MAX HOLDING TIME: Принудительно закрываем если удерживаем слишком долго
        if self.max_holding_bars > 0 and self.position != Positions.FLAT:
            if self.position_holding_bars >= self.max_holding_bars:
                # Принудительно закрываем позицию (время истекло)
                holding_pnl = self._close_position(current_price)
                # Нормализуем reward
                holding_reward = holding_pnl / self.initial_balance
                reward += holding_reward
                reward_breakdown['forced_max_holding_close'] += holding_reward
                self._update_equity(current_price)
        
        # Обновляем статистику
        self._update_statistics()
        
        # Переходим к следующему шагу
        self.current_step += 1
        
        # Проверяем условия завершения
        terminated = self._check_terminated()
        truncated = self.current_step >= len(self.df) - 1
        
        # Принудительное закрытие позиции в конце эпизода
        if truncated and self.position != Positions.FLAT:
            # Закрываем любую открытую позицию по текущей цене
            close_pnl = self._close_position(current_price)
            # Нормализуем reward
            forced_end_reward = close_pnl / self.initial_balance
            reward += forced_end_reward
            reward_breakdown['forced_end_close'] += forced_end_reward
            self._update_equity(current_price)  # Обновляем equity после закрытия
        
        # ТЕРМИНАЛЬНЫЙ БОНУС: добавляем бонус/штраф за общий результат эпизода
        # Это помогает агенту понять связь между действиями и конечным результатом
        if truncated or terminated:
            total_return = (self.equity - self.initial_balance) / self.initial_balance
            # Умеренный масштаб: при return +10% бонус = +1.0
            terminal_bonus = total_return * 10.0

            # DD term отключён (оставляем в breakdown для совместимости логов)
            dd_bonus = 0.0

            reward += terminal_bonus
            reward_breakdown['terminal_bonus'] += terminal_bonus
            reward_breakdown['dd_bonus'] += dd_bonus

            # Optional: discourage "good return but too few trades" regimes.
            # Applies only at episode end; default is disabled.
            if self.too_few_trades_penalty > 0 and self.min_trades_target and self.min_trades_target > 0:
                deficit = int(self.min_trades_target) - int(self.total_trades)
                if deficit > 0:
                    penalty = -self.too_few_trades_penalty * (deficit / float(self.min_trades_target))
                    reward += penalty
                    reward_breakdown['too_few_trades_penalty'] += penalty
        
        # Получаем новое состояние
        observation = self._get_observation()
        info = self._get_info()
        info['reward_breakdown'] = reward_breakdown
        
        # КРИТИЧНО: Убедимся что reward не NaN
        if not np.isfinite(reward):
            reward = 0.0
            reward_breakdown = self._empty_reward_breakdown()
            info['reward_breakdown'] = reward_breakdown

        # Сохраняем breakdown (для eval) и накапливаем суммы по эпизоду
        self.last_reward_breakdown = reward_breakdown
        self.reward_steps += 1
        self.reward_total_sum += float(reward)
        for k, v in reward_breakdown.items():
            self.reward_breakdown_sums[k] += float(v)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, price: float) -> Tuple[float, Dict[str, float]]:
        """Выполняет торговое действие и возвращает награду.

        Упрощённый и более безопасный вариант reward-функции
        для начального обучения:

        - Награда только за *realized* PnL при закрытии позиции
        - Открытие позиции: небольшое отрицательное вознаграждение
          (штраф за издержки + лёгкий анти‑спам)
        - HOLD c позицией: небольшая по модулю промежуточная награда
          по unrealized PnL (dense feedback, но без экстремальных масштабов)
        - HOLD без позиции: маленький, но не огромный штраф
          (чтобы агент не залипал в вечном HOLD)
        """

        reward = 0.0
        breakdown = self._empty_reward_breakdown()

        # SELL Action
        if action == Actions.SELL:
            if self.position == Positions.LONG:
                # Закрываем long — основная награда = realized PnL
                pnl = self._close_position(price)
                reward = pnl / self.initial_balance
                breakdown['realized_close'] += reward
            elif self.position == Positions.FLAT and self.enable_short:
                # Открываем short — минимальный штраф за комиссию
                self._open_short(price)
                reward = -0.0005
                breakdown['open_penalty'] += reward

        # BUY Action
        elif action == Actions.BUY:
            if self.position == Positions.SHORT:
                # Закрываем short — основная награда = realized PnL
                pnl = self._close_position(price)
                reward = pnl / self.initial_balance
                breakdown['realized_close'] += reward
            elif self.position == Positions.FLAT:
                # Открываем long — минимальный штраф за комиссию
                self._open_long(price)
                reward = -0.0005
                breakdown['open_penalty'] += reward

        # HOLD
        elif action == Actions.HOLD:
            if self.position != Positions.FLAT:
                # Есть открытая позиция — симметричный feedback по unrealized PnL
                unrealized_pnl = self._calculate_unrealized_pnl(price)
                normalized = unrealized_pnl / self.initial_balance
                # Симметричный коэффициент для прибыли и убытка
                # Это даёт агенту свободу держать позицию без сильного давления
                reward = 0.3 * normalized
                breakdown['hold_unrealized'] += reward
            else:
                # FLAT и HOLD — очень маленький штраф (не мешает агенту ждать)
                reward = -self.hold_flat_penalty
                breakdown['hold_flat_penalty'] += reward

        return reward, breakdown
    
    def _open_long(self, price: float):
        """Открывает long позицию."""
        position_value = self.balance * self.position_size
        commission_cost = position_value * self.commission
        
        # Покупаем монеты
        self.position_quantity = (position_value - commission_cost) / price
        self.entry_price = price
        self.position = Positions.LONG
        
        # ПРАВИЛЬНО: Вычитаем всю стоимость позиции из баланса
        self.balance -= position_value
        
        self.trade_history.append({
            'step': self.current_step,
            'action': 'OPEN_LONG',
            'price': price,
            'quantity': self.position_quantity,
            'commission': commission_cost
        })

        # Лог сделки для детального анализа
        timestamp = self.df.iloc[self.current_step].get('timestamp') if 'timestamp' in self.df.columns else None
        self.trades_log.append({
            'step': int(self.current_step),
            'timestamp': str(timestamp) if timestamp is not None else '',
            'event': 'OPEN_LONG',
            'price': float(price),
            'quantity': float(self.position_quantity),
            'commission': float(commission_cost),
            'position': 'LONG',
            'pnl': 0.0,
            'equity_after': float(self.equity),
        })
    
    def _open_short(self, price: float):
        """Открывает short позицию."""
        position_value = self.balance * self.position_size
        commission_cost = position_value * self.commission
        
        # Для short мы получаем деньги от продажи (которую ещё не сделали)
        self.position_quantity = (position_value - commission_cost) / price
        self.entry_price = price
        self.position = Positions.SHORT
        
        # SHORT (корректный учет): продаем актив, получаем выручку в cash и платим комиссию.
        # Equity при этом не должна "раздуваться": обязательство выкупить актив отражается
        # отрицательной рыночной стоимостью позиции в _get_position_value().
        proceeds = self.position_quantity * price
        self.balance += (proceeds - commission_cost)
        
        self.trade_history.append({
            'step': self.current_step,
            'action': 'OPEN_SHORT',
            'price': price,
            'quantity': self.position_quantity,
            'commission': commission_cost
        })

        # Лог сделки для детального анализа
        timestamp = self.df.iloc[self.current_step].get('timestamp') if 'timestamp' in self.df.columns else None
        self.trades_log.append({
            'step': int(self.current_step),
            'timestamp': str(timestamp) if timestamp is not None else '',
            'event': 'OPEN_SHORT',
            'price': float(price),
            'quantity': float(self.position_quantity),
            'commission': float(commission_cost),
            'position': 'SHORT',
            'pnl': 0.0,
            'equity_after': float(self.equity),
        })
    
    def _close_position(self, price: float) -> float:
        """Закрывает текущую позицию и возвращает PnL.
        
        Args:
            price: Цена закрытия
        
        Returns:
            Net PnL (profit/loss после комиссий)
        """
        if self.position == Positions.FLAT:
            return 0.0
        
        # Рассчитываем PnL
        if self.position == Positions.LONG:
            pnl = (price - self.entry_price) * self.position_quantity
        else:  # SHORT
            pnl = (self.entry_price - price) * self.position_quantity
        
        # Комиссия за закрытие
        close_value = price * self.position_quantity
        commission_cost = close_value * self.commission
        
        net_pnl = pnl - commission_cost
        
        # ПРАВИЛЬНО: Обновляем баланс
        if self.position == Positions.LONG:
            # LONG: Получаем деньги от продажи минус комиссия
            self.balance += (close_value - commission_cost)
        else:  # SHORT
            # SHORT: Покрываем позицию покупкой актива и платим комиссию.
            self.balance -= (close_value + commission_cost)
        
        # Обновляем статистику
        self.total_profit += net_pnl
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
            self.gross_profit += net_pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(net_pnl)
        
        # Записываем сделку
        self.trade_history.append({
            'step': self.current_step,
            'action': 'CLOSE',
            'price': price,
            'quantity': self.position_quantity,
            'pnl': net_pnl,
            'commission': commission_cost,
            'position_type': 'LONG' if self.position == Positions.LONG else 'SHORT'
        })

        # Лог сделки для детального анализа
        # ВАЖНО: equity_after должен показывать баланс ПОСЛЕ закрытия позиции
        # После закрытия позиция = FLAT, поэтому equity = balance (который уже обновлен выше)
        equity_after_close = self.balance  # balance уже включает PnL от сделки
        
        timestamp = self.df.iloc[self.current_step].get('timestamp') if 'timestamp' in self.df.columns else None
        self.trades_log.append({
            'step': int(self.current_step),
            'timestamp': str(timestamp) if timestamp is not None else '',
            'event': 'CLOSE',
            'price': float(price),
            'quantity': float(self.position_quantity),
            'commission': float(commission_cost),
            'position': 'LONG' if self.position == Positions.LONG else 'SHORT',
            'pnl': float(net_pnl),
            'equity_after': float(equity_after_close),
        })
        
        # Сбрасываем позицию
        self.position = Positions.FLAT
        self.position_quantity = 0.0
        self.entry_price = 0.0
        self.position_holding_bars = 0  # NEW: Reset holding counter (for V4)
        
        return net_pnl
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Рассчитывает unrealized PnL для текущей позиции."""
        if self.position == Positions.LONG:
            return (current_price - self.entry_price) * self.position_quantity
        elif self.position == Positions.SHORT:
            return (self.entry_price - current_price) * self.position_quantity
        return 0.0
    
    def _get_position_value(self, current_price: float) -> float:
        """Рассчитывает текущую рыночную стоимость позиции."""
        if self.position == Positions.LONG:
            # LONG: стоимость = количество * текущая цена
            return self.position_quantity * current_price
        elif self.position == Positions.SHORT:
            # SHORT: позиция — обязательство выкупить актив обратно.
            # Рыночная стоимость обязательства отрицательная.
            return -self.position_quantity * current_price
        return 0.0
    
    def _update_equity(self, current_price: float):
        """Обновляет equity с учетом текущей стоимости позиции."""
        position_value = self._get_position_value(current_price)
        self.equity = self.balance + position_value
        self.equity_history.append(self.equity)
        
        # Обновляем max equity и drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        # Защита от деления на ноль
        if self.max_equity > 0:
            current_dd = (self.max_equity - self.equity) / self.max_equity
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd
        else:
            self.max_drawdown = 0.0
    
    def _update_statistics(self):
        """Обновляет дополнительную статистику."""
        pass  # Можно добавить дополнительные метрики
    
    def _check_terminated(self) -> bool:
        """Проверяет условия принудительного завершения."""
        # ОТКЛЮЧАЕМ early termination - пусть модель учится на длинных эпизодах
        # даже если теряет деньги
        return False
        
        # Закомментировано - было слишком строгое условие
        # if self.equity < self.initial_balance * 0.5:  # -50% = game over
        #     return True
        # return False
    
    def _get_observation(self) -> np.ndarray:
        """Формирует observation для агента."""
        # Получаем окно данных
        window_start = self.current_step - self.lookback_window
        window_end = self.current_step
        
        window_df = self.df.iloc[window_start:window_end].copy()
        
        # Выбираем только feature колонки
        feature_cols = [col for col in window_df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        features = window_df[feature_cols].values
        
        # Добавляем информацию о позиции и equity
        # Рассчитываем unrealized PnL правильно
        if self.position != Positions.FLAT and self.entry_price > 0:
            current_price = self.df.iloc[self.current_step]['close']
            unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * self.position.value
        else:
            unrealized_pnl = 0.0
        
        position_info = np.array([
            self.position.value,
            self.equity / self.initial_balance if self.initial_balance > 0 else 1.0,  # normalized equity
            self.total_profit / self.initial_balance if self.initial_balance > 0 else 0.0,  # normalized profit
            unrealized_pnl,  # unrealized pnl as percentage
            self.max_drawdown
        ])
        
        # Проверяем position_info на NaN/Inf
        position_info = np.nan_to_num(position_info, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Повторяем position_info для каждого временного шага
        position_features = np.tile(position_info, (self.lookback_window, 1))
        
        # Объединяем: features + position_info
        # Multi-timeframe features уже включены в features!
        observation = np.concatenate([features, position_features], axis=1)
        
        # КРИТИЧНО: Конвертируем в float32 СНАЧАЛА
        observation = observation.astype(np.float32)
        
        # Проверяем и очищаем NaN/Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Клипаем для безопасности
        observation = np.clip(observation, -100.0, 100.0)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Возвращает дополнительную информацию о состоянии."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else 0.0
        
        # Процентный возврат
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        total_return_pct = total_return * 100
        
        # Sharpe Ratio (упрощенный)
        if len(self.equity_history) > 1:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Аннуализированный
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max Drawdown в процентах
        max_drawdown_pct = self.max_drawdown * 100
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position.name,
            'total_profit': self.total_profit,
            'total_return_pct': total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'return': total_return
        }
    
    def render(self, mode='human'):
        """Визуализация состояния (опционально)."""
        info = self._get_info()
        print(f"\nStep: {info['step']}")
        print(f"Equity: ${info['equity']:.2f} (Return: {info['return']*100:.2f}%)")
        print(f"Position: {info['position']}")
        print(f"Trades: {info['total_trades']} (WR: {info['win_rate']*100:.1f}%)")
        print(f"Max DD: {info['max_drawdown']*100:.2f}%")
    
    def get_trade_history(self) -> pd.DataFrame:
        """Возвращает историю сделок."""
        return pd.DataFrame(self.trade_history)


# Пример использования
if __name__ == "__main__":
    from feature_engineering import FeatureEngineer
    from data_loader import DataLoader
    
    # Загружаем данные
    loader = DataLoader()
    df = loader.load_data('BTCUSDT', timeframe='30m', days=90)
    
    # Создаем фичи
    engineer = FeatureEngineer()
    df_features = engineer.calculate_features(df)
    
    # Создаем окружение
    env = MarketTradingEnv(df_features, lookback_window=60)
    
    print("🎮 Testing Trading Environment...")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    
    # Случайные действия для теста
    obs, info = env.reset()
    print(f"\nInitial state: {info}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode finished at step {i}")
            break
    
    print(f"\nFinal info: {info}")
    print(f"\nTrade history:\n{env.get_trade_history()}")
