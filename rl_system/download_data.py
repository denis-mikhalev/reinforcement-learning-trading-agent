"""
Утилита для предварительной загрузки данных с Binance
Использование: python download_data.py --symbol BTCUSDT --timeframe 30m --days 90
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Загрузка данных с Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Торговая пара")
    parser.add_argument("--timeframe", type=str, default="30m", help="Таймфрейм")
    parser.add_argument("--days", type=int, default=90, help="Количество дней истории")
    
    args = parser.parse_args()
    
    print(f"🌐 Загрузка {args.symbol} {args.timeframe} за {args.days} дней...")
    
    loader = DataLoader()
    df = loader.load_data(args.symbol, args.timeframe, args.days)
    
    print(f"✅ Загружено {len(df)} свечей")
    print(f"📅 Период: {df.index[0]} - {df.index[-1]}")
    print(f"💾 Сохранено в: data/{args.symbol}_{args.timeframe}_{args.days}d.csv")


if __name__ == "__main__":
    main()
