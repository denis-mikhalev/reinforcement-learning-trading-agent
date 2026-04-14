"""
Data Loader для загрузки исторических данных с Binance
=======================================================

Поддерживает кеширование, несколько таймфреймов, мультисимвольную загрузку.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from binance.client import Client


class DataLoader:
    """
    Загружает и кеширует исторические данные с Binance.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 cache_dir: str = "rl_system/data"):
        """
        Args:
            api_key: Binance API key (опционально, для публичных данных не нужен)
            api_secret: Binance API secret
            cache_dir: Директория для кеша
        """
        # Откладываем создание клиента до первого использования (ленивая загрузка)
        self._api_key = api_key or ""
        self._api_secret = api_secret or ""
        self._client = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Ленивая инициализация Binance клиента с fallback на Binance.US."""
        if self._client is None:
            try:
                # Пробуем стандартный Binance API
                self._client = Client(self._api_key, self._api_secret)
                print("✅ Подключение к Binance.com")
            except Exception as e:
                # Если ошибка связана с geo-ограничениями, переключаемся на Binance.US
                error_msg = str(e)
                if "restricted location" in error_msg.lower() or "service unavailable" in error_msg.lower():
                    print(f"⚠️  Binance.com недоступен (geo-ограничение)")
                    print("🔄 Переключение на Binance.US...")
                    
                    try:
                        # Создаем клиент для Binance.US используя параметр tld='us'
                        self._client = Client(self._api_key, self._api_secret, tld='us')
                        print("✅ Подключение к Binance.US (api.binance.us)")
                    except Exception as us_error:
                        print(f"❌ Binance.US также недоступен: {us_error}")
                        raise RuntimeError(f"Не удалось подключиться ни к Binance.com, ни к Binance.US: {e}")
                else:
                    # Другая ошибка - пробрасываем дальше
                    raise
        return self._client
    
    def load_data(self,
                  symbol: str,
                  timeframe: str = '30m',
                  days: int = 365,
                  use_cache: bool = True,
                  max_cache_age_hours: int = 24) -> pd.DataFrame:
        """
        Загружает исторические данные.
        
        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            timeframe: Таймфрейм ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            days: Количество дней истории
            use_cache: Использовать кеш
            max_cache_age_hours: Максимальный возраст кэша в часах (по умолчанию 24)
        
        Returns:
            DataFrame с колонками ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        cache_file = self._get_cache_filename(symbol, timeframe, days)
        
        # Проверяем кеш
        if use_cache and cache_file.exists():
            # Проверяем возраст кэша
            cache_age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
            
            if cache_age_hours < max_cache_age_hours:
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    
                    # Проверяем актуальность данных
                    last_data_date = pd.to_datetime(df['timestamp'].iloc[-1])
                    hours_since_last_data = (datetime.now() - last_data_date.to_pydatetime()).total_seconds() / 3600
                    
                    print(f"📦 Загружено из кеша: {symbol} {timeframe} ({len(df)} баров)")
                    print(f"   Последняя свеча: {last_data_date.strftime('%Y-%m-%d %H:%M')} ({hours_since_last_data:.1f}ч назад)")
                    
                    # Если данные свежие - возвращаем
                    if hours_since_last_data < max_cache_age_hours:
                        return df
                    else:
                        print(f"   ⚠️  Данные устарели ({hours_since_last_data:.1f}ч > {max_cache_age_hours}ч), обновляем...")
                except Exception as e:
                    print(f"⚠️ Ошибка чтения кеша: {e}")
            else:
                print(f"⚠️ Кэш устарел ({cache_age_hours:.1f}ч > {max_cache_age_hours}ч), обновляем...")
        
        # Загружаем с Binance
        print(f"🌐 Загрузка {symbol} {timeframe} за последние {days} дней...")
        df = self._fetch_from_binance(symbol, timeframe, days)
        
        # Сохраняем в кеш
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"💾 Сохранено в кеш: {cache_file.name}")
            except Exception as e:
                print(f"⚠️ Ошибка сохранения кеша: {e}")
        
        return df
    
    def _fetch_from_binance(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Загружает данные с Binance API."""
        # Рассчитываем временной интервал
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Конвертируем в timestamp
        start_str = start_time.strftime('%d %b %Y')
        
        # Загружаем klines
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=start_str
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки данных с Binance: {e}")
        
        # Конвертируем в DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Оставляем только нужные колонки
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Конвертируем типы
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        print(f"✅ Загружено {len(df)} баров ({df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]})")
        
        return df
    
    def load_multiple_symbols(self,
                             symbols: List[str],
                             timeframe: str = '30m',
                             days: int = 365,
                             use_cache: bool = True) -> dict:
        """
        Загружает данные для нескольких символов.
        
        Returns:
            Dict[symbol, DataFrame]
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.load_data(symbol, timeframe, days, use_cache)
                data[symbol] = df
            except Exception as e:
                print(f"❌ Ошибка загрузки {symbol}: {e}")
        
        return data
    
    def split_train_test(self,
                        df: pd.DataFrame,
                        train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделяет данные на train/test по времени.
        
        Args:
            df: DataFrame с данными
            train_ratio: Доля train данных (0.0 - 1.0)
        
        Returns:
            (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"📊 Train: {len(train_df)} баров ({train_df['timestamp'].iloc[0]} - {train_df['timestamp'].iloc[-1]})")
        print(f"📊 Test: {len(test_df)} баров ({test_df['timestamp'].iloc[0]} - {test_df['timestamp'].iloc[-1]})")
        
        return train_df, test_df
    
    def _get_cache_filename(self, symbol: str, timeframe: str, days: int) -> Path:
        """Генерирует имя файла для кеша."""
        return self.cache_dir / f"{symbol}_{timeframe}_{days}d.pkl"
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Очищает кеш.
        
        Args:
            symbol: Если указан, удаляет только кеш для этого символа
        """
        if symbol:
            pattern = f"{symbol}_*.pkl"
        else:
            pattern = "*.pkl"
        
        removed = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            removed += 1
        
        print(f"🗑️ Удалено {removed} файлов кеша")
    
    def get_market_info(self, symbol: str) -> dict:
        """Получает информацию о торговой паре."""
        try:
            info = self.client.get_symbol_info(symbol)
            return {
                'symbol': info['symbol'],
                'status': info['status'],
                'base_asset': info['baseAsset'],
                'quote_asset': info['quoteAsset'],
                'price_precision': info['quotePrecision'],
                'quantity_precision': info['baseAssetPrecision']
            }
        except Exception as e:
            print(f"❌ Ошибка получения информации о {symbol}: {e}")
            return {}


# Пример использования
if __name__ == "__main__":
    # Создаем загрузчик
    loader = DataLoader()
    
    # Загружаем данные для BTC
    df = loader.load_data('BTCUSDT', timeframe='30m', days=180)
    print(f"\nЗагружено данных: {df.shape}")
    print(f"\nПервые строки:\n{df.head()}")
    print(f"\nПоследние строки:\n{df.tail()}")
    
    # Разделяем на train/test
    train_df, test_df = loader.split_train_test(df, train_ratio=0.8)
    
    # Получаем информацию о паре
    info = loader.get_market_info('BTCUSDT')
    print(f"\nИнформация о паре:\n{info}")
    
    # Загружаем несколько символов
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    multi_data = loader.load_multiple_symbols(symbols, timeframe='1h', days=90)
    print(f"\nЗагружено символов: {len(multi_data)}")
    for symbol, df in multi_data.items():
        print(f"  {symbol}: {len(df)} баров")
