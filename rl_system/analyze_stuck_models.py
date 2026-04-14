"""
Анализатор "застрявших" RL моделей
====================================

Сканирует все модели из live_signals_summary и выводит детальную информацию
о моделях, которые застряли на одном типе сигнала более N дней.

Использование:
    python rl_system/analyze_stuck_models.py
    python rl_system/analyze_stuck_models.py --threshold 14  # для 14 дней
    python rl_system/analyze_stuck_models.py --symbol ADAUSDT
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl_system.live_signals_summary import (
    load_latest_states,
    _get_model_stats,
    _check_if_model_stuck,
)


def analyze_stuck_models(symbol_filters=None, threshold_days=7):
    """Анализирует все модели и выводит список застрявших."""
    
    by_symbol = load_latest_states(symbol_filters=symbol_filters)
    now_dt = datetime.now()
    
    stuck_models = []
    total_models = 0
    
    for symbol, signals in by_symbol.items():
        for signal_type in ["BUY", "SELL", "HOLD"]:
            for model_entry in signals[signal_type]:
                total_models += 1
                name = model_entry.get("name", "<unknown>")
                timeframe = model_entry.get("timeframe")
                
                stats = _get_model_stats(symbol, timeframe, name, now=now_dt)
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=threshold_days)
                
                if stuck_signal and stuck_signal != False:
                    # Модель застряла
                    stuck_info = {
                        "name": name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "current_signal": signal_type,
                        "stuck_on": stuck_signal,
                        "stats": stats,
                    }
                    stuck_models.append(stuck_info)
    
    return stuck_models, total_models


def format_stuck_report(stuck_models, total_models, threshold_days):
    """Форматирует отчет о застрявших моделях."""
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"ОТЧЕТ О ЗАСТРЯВШИХ МОДЕЛЯХ (threshold: {threshold_days} дней)")
    lines.append("=" * 80)
    lines.append("")
    
    if not stuck_models:
        lines.append(f"✅ Отлично! Все {total_models} моделей работают нормально.")
        lines.append("   Ни одна модель не застряла на одном сигнале более {threshold_days} дней.")
        return "\n".join(lines)
    
    stuck_pct = (len(stuck_models) / total_models * 100) if total_models > 0 else 0
    lines.append(f"⚠️  Найдено застрявших моделей: {len(stuck_models)} из {total_models} ({stuck_pct:.1f}%)")
    lines.append("")
    
    # Группируем по символу
    by_symbol = defaultdict(list)
    for model in stuck_models:
        by_symbol[model["symbol"]].append(model)
    
    for symbol in sorted(by_symbol.keys()):
        models = by_symbol[symbol]
        lines.append(f"📊 {symbol} — {len(models)} застрявших моделей:")
        lines.append("")
        
        for m in sorted(models, key=lambda x: x["name"]):
            name = m["name"]
            stuck_on = m["stuck_on"]
            current = m["current_signal"]
            stats = m["stats"]
            
            # Извлекаем статистику за 7 и 30 дней
            s7d = stats.get("7d", {}) if stats else {}
            s30d = stats.get("30d", {}) if stats else {}
            
            b7 = s7d.get("BUY", 0)
            s7 = s7d.get("SELL", 0)
            h7 = s7d.get("HOLD", 0)
            total_7d = b7 + s7 + h7
            
            b30 = s30d.get("BUY", 0)
            s30 = s30d.get("SELL", 0)
            h30 = s30d.get("HOLD", 0)
            total_30d = b30 + s30 + h30
            
            lines.append(f"   ⚠️  {name}")
            lines.append(f"      Застрял на: {stuck_on}")
            lines.append(f"      Текущий сигнал: {current}")
            lines.append(f"      7d:  B={b7}/S={s7}/H={h7} (total={total_7d})")
            lines.append(f"      30d: B={b30}/S={s30}/H={h30} (total={total_30d})")
            lines.append("")
    
    lines.append("=" * 80)
    lines.append("РЕКОМЕНДАЦИИ:")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. Проверьте логи обучения этих моделей (возможен overfit)")
    lines.append("2. Рассмотрите переобучение с:")
    lines.append("   - Добавлением entropy bonus в reward функцию")
    lines.append("   - Более разнообразными тренировочными данными")
    lines.append("   - Penalty за однообразие действий")
    lines.append("3. Если модель застряла > 14 дней — удалите из registry")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Анализирует 'застрявшие' RL модели"
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="Фильтр по символу (например ADAUSDT). Можно указать несколько раз.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help="Порог в днях для определения 'застрявшей' модели (по умолчанию: 7)",
    )
    parser.add_argument(
        "--export-json",
        help="Экспортировать результаты в JSON файл",
    )
    
    args = parser.parse_args()
    
    symbols = args.symbol if args.symbol else None
    
    print("🔍 Анализирую модели...")
    stuck_models, total_models = analyze_stuck_models(
        symbol_filters=symbols,
        threshold_days=args.threshold
    )
    
    report = format_stuck_report(stuck_models, total_models, args.threshold)
    print(report)
    
    if args.export_json and stuck_models:
        export_path = Path(args.export_json)
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "threshold_days": args.threshold,
            "total_models": total_models,
            "stuck_models_count": len(stuck_models),
            "stuck_models": stuck_models,
        }
        
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Экспортировано в: {export_path}")
        except Exception as e:
            print(f"\n❌ Ошибка экспорта: {e}")


if __name__ == "__main__":
    main()
