"""
Утилита для очистки кэша данных
================================

Использование:
--------------
# Очистить весь кэш
python rl_system/clear_data_cache.py

# Очистить кэш для конкретного символа
python rl_system/clear_data_cache.py --symbol BTCUSDT

# Очистить кэш для конкретного таймфрейма
python rl_system/clear_data_cache.py --symbol ADAUSDT --timeframe 1h

# Показать информацию о кэше
python rl_system/clear_data_cache.py --list
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from data_loader import DataLoader


def format_size(bytes_size):
    """Форматирует размер в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_age(timestamp):
    """Форматирует возраст файла."""
    age_seconds = datetime.now().timestamp() - timestamp
    age_hours = age_seconds / 3600
    
    if age_hours < 1:
        return f"{age_seconds/60:.0f}m"
    elif age_hours < 24:
        return f"{age_hours:.1f}h"
    else:
        age_days = age_hours / 24
        return f"{age_days:.1f}d"


def list_cache(cache_dir="data/cache"):
    """Показывает информацию о кэше."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    cache_files = list(cache_path.glob("*.pkl"))
    
    if not cache_files:
        print(f"✅ No cache files found in {cache_dir}")
        return
    
    print(f"\n📦 Cache files in {cache_dir}:")
    print(f"{'File':<40} {'Size':<10} {'Age':<10} {'Modified':<20}")
    print("=" * 85)
    
    total_size = 0
    for cache_file in sorted(cache_files):
        size = cache_file.stat().st_size
        mtime = cache_file.stat().st_mtime
        age = format_age(mtime)
        modified = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{cache_file.name:<40} {format_size(size):<10} {age:<10} {modified:<20}")
        total_size += size
    
    print("=" * 85)
    print(f"Total: {len(cache_files)} files, {format_size(total_size)}")
    print()


def clear_cache(symbol=None, timeframe=None, cache_dir="data/cache"):
    """Очищает кэш."""
    loader = DataLoader(cache_dir=cache_dir)
    
    if symbol and timeframe:
        pattern = f"{symbol}_{timeframe}_*.pkl"
        description = f"{symbol} {timeframe}"
    elif symbol:
        pattern = f"{symbol}_*.pkl"
        description = symbol
    else:
        pattern = "*.pkl"
        description = "all symbols"
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"✅ No cache directory found")
        return
    
    cache_files = list(cache_path.glob(pattern))
    
    if not cache_files:
        print(f"✅ No cache files found for {description}")
        return
    
    total_size = sum(f.stat().st_size for f in cache_files)
    
    print(f"\n🗑️  Clearing cache for {description}...")
    print(f"   Files to remove: {len(cache_files)}")
    print(f"   Total size: {format_size(total_size)}")
    
    for cache_file in cache_files:
        print(f"   Removing: {cache_file.name}")
        cache_file.unlink()
    
    print(f"✅ Cleared {len(cache_files)} cache files ({format_size(total_size)})")


def main():
    parser = argparse.ArgumentParser(
        description='Manage data cache for RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                       help='List cache files')
    parser.add_argument('--symbol', '-s', type=str,
                       help='Clear cache for specific symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', '-t', type=str,
                       help='Clear cache for specific timeframe (e.g., 1h)')
    parser.add_argument('--cache-dir', type=str, default='data/cache',
                       help='Cache directory (default: data/cache)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation')
    
    args = parser.parse_args()
    
    if args.list:
        list_cache(args.cache_dir)
    else:
        # Подтверждение
        if not args.yes:
            if args.symbol:
                msg = f"Clear cache for {args.symbol}"
                if args.timeframe:
                    msg += f" {args.timeframe}"
            else:
                msg = "Clear ALL cache"
            
            response = input(f"{msg}? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Cancelled")
                return
        
        clear_cache(args.symbol, args.timeframe, args.cache_dir)
        print("\n💡 Tip: Use --list to view remaining cache files")


if __name__ == "__main__":
    main()
