"""
Feature Engineering для RL Trading System
==========================================

Рассчитывает технические индикаторы и подготавливает данные для агента.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import talib


class FeatureEngineer:
    """
    Класс для расчета технических индикаторов и подготовки фичей.
    
    Индикаторы:
    - Трендовые: SMA, EMA, MACD
    - Momentum: RSI, Stochastic, ROC
    - Volatility: ATR, Bollinger Bands
    - Volume: OBV, Volume SMA
    - Price Action: Returns, High/Low ranges
    """
    
    def __init__(self, lookback_periods: Optional[List[int]] = None):
        """
        Args:
            lookback_periods: Периоды для расчета индикаторов (по умолчанию: [7, 14, 21, 50])
        """
        self.lookback_periods = lookback_periods or [7, 14, 21, 50]
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все технические индикаторы.
        
        Args:
            df: DataFrame с колонками ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame с добавленными фичами
        """
        df = df.copy()
        
        # 1. Price Action Features
        df = self._add_price_action_features(df)
        
        # 2. Trend Indicators
        df = self._add_trend_indicators(df)
        
        # 3. Momentum Indicators
        df = self._add_momentum_indicators(df)
        
        # 4. Volatility Indicators
        df = self._add_volatility_indicators(df)
        
        # 5. Volume Indicators (ENHANCED)
        df = self._add_volume_indicators(df)
        
        # 6. Market Structure
        df = self._add_market_structure(df)
        
        # 7. Volume Profile Features (NEW)
        df = self._add_volume_profile_features(df)
        
        # 8. Order Flow Proxies (NEW)
        df = self._add_order_flow_features(df)
        
        # 9. Market Regime Features (NEW)
        df = self._add_market_regime_features(df)
        
        # Очистка данных: убираем NaN и Inf
        # 1. Заменяем Inf на NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Убираем строки с NaN
        df = df.dropna().reset_index(drop=True)
        
        # 3. Клипаем экстремальные значения для численной стабильности
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols:
            # Клипаем на уровне 99.9 перцентиля
            lower_bound = df[col].quantile(0.001)
            upper_bound = df[col].quantile(0.999)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет фичи на основе price action."""
        # Returns на разных таймфреймах
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close range
        df['oc_range'] = (df['close'] - df['open']) / df['open']
        
        # Upper/Lower shadows (wick size)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Body size
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет трендовые индикаторы."""
        close = df['close'].values
        
        # Moving Averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # Distance from MA (normalized)
            df[f'dist_sma_{period}'] = (close - df[f'sma_{period}']) / close
            df[f'dist_ema_{period}'] = (close - df[f'ema_{period}']) / close
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, 
                                                    fastperiod=12, 
                                                    slowperiod=26, 
                                                    signalperiod=9)
        df['macd'] = macd / close  # normalized
        df['macd_signal'] = macd_signal / close
        df['macd_hist'] = macd_hist / close
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, close, timeperiod=14)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет momentum индикаторы."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI на разных периодах
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close,
                                    fastk_period=14,
                                    slowk_period=3,
                                    slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Rate of Change
        for period in [7, 14, 21]:
            df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
        
        # Commodity Channel Index
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет индикаторы волатильности."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period) / close
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle  # Normalized width
        df['bb_position'] = (close - lower) / (upper - lower)  # Position within bands
        
        # Historical Volatility
        for period in [7, 14, 21]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет индикаторы объема."""
        close = df['close'].values
        volume = df['volume'].values
        
        # Volume SMA
        for period in [7, 14, 21]:
            df[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
            df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']
        
        # OBV (On-Balance Volume)
        df['obv'] = talib.OBV(close, volume)
        df['obv_sma'] = talib.SMA(df['obv'].values, timeperiod=14)
        
        # Volume-Price Trend
        df['vpt'] = ((close - np.roll(close, 1)) / np.roll(close, 1) * volume).cumsum()
        
        return df
    
    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет структурные фичи рынка."""
        close = df['close'].values
        
        # Higher highs / Lower lows
        for period in [7, 14]:
            df[f'higher_high_{period}'] = (close > df['high'].rolling(period).max().shift(1)).astype(int)
            df[f'lower_low_{period}'] = (close < df['low'].rolling(period).min().shift(1)).astype(int)
        
        # Distance from recent high/low
        for period in [14, 21, 50]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            df[f'dist_from_high_{period}'] = (close - rolling_high) / close
            df[f'dist_from_low_{period}'] = (close - rolling_low) / close
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Нормализует фичи.
        
        Args:
            df: DataFrame с фичами
            method: 'minmax' или 'zscore'
        
        Returns:
            Нормализованный DataFrame
        """
        df = df.copy()
        
        # Колонки для нормализации (все кроме OHLCV и временных меток)
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        if method == 'minmax':
            # Min-Max нормализация [0, 1]
            for col in feature_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0
        
        elif method == 'zscore':
            # Z-score нормализация
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Возвращает список названий фичей (без OHLCV)."""
        return [col for col in df.columns 
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def _add_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет фичи на основе volume profile.
        Volume profile показывает распределение объема по ценовым уровням.
        """
        # Volume-weighted features
        df['vwap'] = (df['close'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
        df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']
        
        # Volume momentum
        for period in [7, 14, 21]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        df['vpt_sma_14'] = df['vpt'].rolling(window=14).mean()
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        df['mfi'] = mfi
        
        # Volume-weighted momentum
        df['volume_momentum_7'] = (df['close'].pct_change(7) * df['volume']).rolling(7).sum()
        df['volume_momentum_14'] = (df['close'].pct_change(14) * df['volume']).rolling(14).sum()
        
        return df
    
    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет прокси для order flow (направления ордеров).
        В отсутствие реальных данных orderbook используем аппроксимации.
        """
        # Buy/Sell pressure approximation
        # Если close > open - больше покупок, иначе - продаж
        df['buy_pressure'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)).clip(-1, 1)
        
        # Volume-weighted buy pressure
        df['weighted_buy_pressure'] = df['buy_pressure'] * df['volume']
        df['buy_pressure_sma_7'] = df['weighted_buy_pressure'].rolling(7).sum() / df['volume'].rolling(7).sum()
        df['buy_pressure_sma_14'] = df['weighted_buy_pressure'].rolling(14).sum() / df['volume'].rolling(14).sum()
        
        # Aggressive buying/selling (price moves with high volume)
        df['aggressive_buying'] = ((df['close'] > df['open']) & (df['volume'] > df['volume'].rolling(14).mean())).astype(int)
        df['aggressive_selling'] = ((df['close'] < df['open']) & (df['volume'] > df['volume'].rolling(14).mean())).astype(int)
        
        df['aggressive_buy_ratio_7'] = df['aggressive_buying'].rolling(7).sum() / 7
        df['aggressive_sell_ratio_7'] = df['aggressive_selling'].rolling(7).sum() / 7
        
        # Delta approximation (net buying/selling pressure)
        # Close near high = buying, close near low = selling
        df['delta'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) - 0.5
        df['delta_sma_7'] = df['delta'].rolling(7).mean()
        df['delta_sma_14'] = df['delta'].rolling(14).mean()
        
        # Cumulative delta (накопленное давление)
        df['cumulative_delta'] = (df['delta'] * df['volume']).cumsum()
        df['cumulative_delta_change_14'] = df['cumulative_delta'].pct_change(14)
        
        # Volume imbalance (большой объем на одной свече = дисбаланс)
        df['volume_imbalance'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume_imbalance'] > 2).astype(int)
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет фичи для определения рыночного режима (trending/ranging/volatile).
        """
        close = df['close'].values
        
        # Trend strength (ADX already calculated in trend indicators)
        # High ADX = strong trend, Low ADX = ranging
        
        # Volatility regime
        df['volatility_7'] = df['close'].pct_change().rolling(7).std()
        df['volatility_21'] = df['close'].pct_change().rolling(21).std()
        df['volatility_ratio'] = df['volatility_7'] / (df['volatility_21'] + 1e-10)
        
        # Choppiness Index (определяет ranging market)
        # Используем atr_14 вместо atr
        atr_sum = df['atr_14'].rolling(14).sum()
        high_low_range = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['choppiness'] = 100 * np.log10(atr_sum / (high_low_range + 1e-10)) / np.log10(14)
        
        # Efficiency Ratio (Kaufman's) - насколько эффективно движется цена
        price_change = abs(df['close'] - df['close'].shift(14))
        volatility_sum = abs(df['close'].diff()).rolling(14).sum()
        df['efficiency_ratio'] = price_change / (volatility_sum + 1e-10)
        
        # Higher Timeframe Trend (аппроксимация старшего таймфрейма)
        # Используем более длинные MA для определения общего тренда
        df['htf_trend'] = (df['ema_50'] > df['sma_50']).astype(int)  # 1 = uptrend, 0 = downtrend
        df['htf_trend_strength'] = abs(df['close'] - df['sma_50']) / df['close']
        
        # Price distance from key levels
        df['dist_from_high_50'] = (df['high'].rolling(50).max() - df['close']) / df['close']
        df['dist_from_low_50'] = (df['close'] - df['low'].rolling(50).min()) / df['close']
        
        # Market phase (accumulation/markup/distribution/markdown)
        # Based on volume and price action
        # Отключаем PerformanceWarning о "highly fragmented" DataFrame,
        # который не критичен для работы live‑системы, но засоряет консоль.
        import warnings
        from pandas.errors import PerformanceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)

            volume_ma = df['volume'].rolling(20).mean()
            price_change_20 = df['close'].pct_change(20)
            
            # Accumulation: low volume, small price change
            df['accumulation_score'] = ((df['volume'] < volume_ma) & (abs(price_change_20) < 0.05)).astype(int)
            
            # Distribution: high volume, small price change
            df['distribution_score'] = ((df['volume'] > volume_ma) & (abs(price_change_20) < 0.05)).astype(int)
            
            # Markup: high volume, positive price change
            df['markup_score'] = ((df['volume'] > volume_ma) & (price_change_20 > 0.05)).astype(int)
            
            # Markdown: high volume, negative price change
            df['markdown_score'] = ((df['volume'] > volume_ma) & (price_change_20 < -0.05)).astype(int)
        
        return df
    
    def add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет Weekly и Monthly features для долгосрочного контекста.
        
        Дает агенту "зум аут" - видеть долгосрочные тренды без увеличения lookback.
        
        Args:
            df: DataFrame с daily данными и фичами
            
        Returns:
            DataFrame с добавленными weekly/monthly features
        """
        df = df.copy()
        
        # Сохраняем timestamp как datetime для resampling
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 1. WEEKLY FEATURES (последние 52 недели = 1 год долгосрочного контекста)
        # Логи выключены, чтобы не засорять консоль при live‑запуске
        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Weekly indicators (только ключевые)
        weekly['weekly_rsi'] = talib.RSI(weekly['close'], timeperiod=14)
        weekly['weekly_macd'], weekly['weekly_macd_signal'], _ = talib.MACD(
            weekly['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        weekly['weekly_ema20'] = talib.EMA(weekly['close'], timeperiod=20)
        weekly['weekly_ema50'] = talib.EMA(weekly['close'], timeperiod=50)
        weekly['weekly_atr'] = talib.ATR(weekly['high'], weekly['low'], weekly['close'], timeperiod=14)
        
        # Weekly trend direction
        weekly['weekly_trend'] = np.where(weekly['close'] > weekly['weekly_ema20'], 1, 
                                         np.where(weekly['close'] < weekly['weekly_ema20'], -1, 0))
        
        # Merge weekly features обратно в daily (forward fill)
        weekly_cols = ['weekly_rsi', 'weekly_macd', 'weekly_macd_signal', 
                      'weekly_ema20', 'weekly_ema50', 'weekly_atr', 'weekly_trend']
        
        for col in weekly_cols:
            df[col] = weekly[col].reindex(df.index, method='ffill')
        
        # 2. MONTHLY FEATURES (последние 12-24 месяца = multi-year контекст)
        # Логи выключены, чтобы не засорять консоль при live‑запуске
        monthly = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Monthly indicators (еще более долгосрочные)
        monthly['monthly_rsi'] = talib.RSI(monthly['close'], timeperiod=14)
        monthly['monthly_ema12'] = talib.EMA(monthly['close'], timeperiod=12)
        monthly['monthly_ema24'] = talib.EMA(monthly['close'], timeperiod=24)
        
        # Monthly trend (сильный долгосрочный сигнал)
        monthly['monthly_trend'] = np.where(monthly['close'] > monthly['monthly_ema12'], 1,
                                           np.where(monthly['close'] < monthly['monthly_ema12'], -1, 0))
        
        # Monthly momentum
        monthly['monthly_momentum'] = monthly['close'].pct_change(3)  # 3-month momentum
        
        # Merge monthly features обратно в daily
        monthly_cols = ['monthly_rsi', 'monthly_ema12', 'monthly_ema24', 
                       'monthly_trend', 'monthly_momentum']
        
        for col in monthly_cols:
            df[col] = monthly[col].reindex(df.index, method='ffill')
        
        # 3. TIMEFRAME ALIGNMENT FEATURES
        # Показывает согласованность между timeframes (сильный сигнал!)
        df['trend_alignment'] = (
            (df.get('trend_direction', 0) * 0.4) +  # Daily: 40% weight
            (df['weekly_trend'] * 0.4) +             # Weekly: 40% weight  
            (df['monthly_trend'] * 0.2)              # Monthly: 20% weight
        )
        
        # RSI alignment (все timeframes в overbought/oversold?)
        df['rsi_alignment'] = (
            ((df.get('rsi_14', 50) - 50) / 50 * 0.5) +       # Daily RSI normalized
            ((df['weekly_rsi'] - 50) / 50 * 0.3) +           # Weekly RSI
            ((df['monthly_rsi'] - 50) / 50 * 0.2)            # Monthly RSI
        )
        
        # Reset index back to column
        df.reset_index(inplace=True)
        
        # Fill NaN values (first weeks/months won't have data)
        multi_tf_cols = weekly_cols + monthly_cols + ['trend_alignment', 'rsi_alignment']
        for col in multi_tf_cols:
            if col in df.columns:
                df[col] = df[col].bfill().fillna(0)
        
        # Информационный лог отключён, чтобы не спамить консоль при каждом запуске
        
        return df


# Пример использования
if __name__ == "__main__":
    # Генерируем тестовые данные
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    np.random.seed(42)
    
    price = 100
    prices = [price]
    for _ in range(499):
        price = price * (1 + np.random.randn() * 0.02)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    # Создаем фичи
    engineer = FeatureEngineer()
    df_features = engineer.calculate_features(df)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")
    print(f"\nFeature names: {engineer.get_feature_names(df_features)[:10]}...")
    print(f"\nDataFrame shape: {df_features.shape}")
    print(f"\nSample data:\n{df_features.tail()}")
