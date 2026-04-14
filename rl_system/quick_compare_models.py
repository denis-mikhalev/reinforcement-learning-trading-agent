"""
Быстрое сравнение нескольких RL моделей
==========================================

Читает best/config.json и root/config.json для каждой модели
и выводит сравнительную таблицу с метриками + диагностикой переобучения.
"""

import json
from pathlib import Path
import sys

def load_model_metrics(model_path):
    """Загружает метрики BEST и FINAL модели."""
    model_path = Path(model_path)
    
    # Загружаем BEST config
    best_config_path = model_path / 'best' / 'config.json'
    if not best_config_path.exists():
        return None
    
    with open(best_config_path, 'r') as f:
        best_config = json.load(f)
    
    # Загружаем FINAL config
    final_config_path = model_path / 'config.json'
    if not final_config_path.exists():
        return None
    
    with open(final_config_path, 'r') as f:
        final_config = json.load(f)
    
    best_results = best_config.get('evaluation_results', {})
    final_results = final_config.get('evaluation_results', {})
    
    return {
        'name': model_path.name,
        'algorithm': best_config.get('algorithm', 'Unknown'),
        'timeframe': best_config.get('timeframe', 'Unknown'),
        'timesteps': best_config.get('total_timesteps', 0),
        'best_return': best_results.get('final_return_pct', 0),
        'best_trades': best_results.get('total_trades', 0),
        'best_profit_factor': best_results.get('profit_factor', 0),
        'best_win_rate': best_results.get('win_rate_pct', 0),
        'best_sharpe': best_results.get('sharpe_ratio', 0),
        'best_max_dd': best_results.get('max_drawdown_pct', 0),
        'final_return': final_results.get('final_return_pct', 0),
        'overfitting': abs(best_results.get('final_return_pct', 0) - final_results.get('final_return_pct', 0))
    }

def get_overfitting_status(overfitting):
    """Возвращает статус переобучения."""
    if overfitting < 3:
        return "✅ Stable"
    elif overfitting < 10:
        return "⚠️  Moderate"
    else:
        return "❌ Strong"

def print_comparison_table(models):
    """Выводит таблицу сравнения моделей."""
    print("\n" + "="*140)
    print("🔍 RL Models Comparison")
    print("="*140)
    
    # Заголовок
    print(f"{'Model':<35} {'Algo':<6} {'TF':<5} {'Steps':<8} {'Return':<9} {'PF':<6} {'Trades':<8} {'WinRate':<9} {'Sharpe':<8} {'MaxDD':<8} {'Overfit':<12}")
    print("-"*140)
    
    # Сортируем по return (лучшие сверху)
    models_sorted = sorted(models, key=lambda x: x['best_return'], reverse=True)
    
    for m in models_sorted:
        overfit_status = get_overfitting_status(m['overfitting'])
        
        print(
            f"{m['name']:<35} "
            f"{m['algorithm']:<6} "
            f"{m['timeframe']:<5} "
            f"{m['timesteps']:<8,} "
            f"{m['best_return']:>+7.2f}% "
            f"{m['best_profit_factor']:>5.2f} "
            f"{m['best_trades']:>7} "
            f"{m['best_win_rate']:>7.1f}% "
            f"{m['best_sharpe']:>7.2f} "
            f"{m['best_max_dd']:>7.1f}% "
            f"{m['overfitting']:>6.2f}% {overfit_status}"
        )
    
    print("="*140)
    
    # Находим лучшую модель
    if models_sorted:
        best = models_sorted[0]
        print(f"\n🏆 Best Model: {best['name']}")
        print(f"   Algorithm: {best['algorithm']} ({best['timeframe']})")
        print(f"   Return: {best['best_return']:+.2f}%")
        print(f"   Profit Factor: {best['best_profit_factor']:.2f}")
        print(f"   Trades: {best['best_trades']} (Win Rate: {best['best_win_rate']:.1f}%)")
        print(f"   Overfitting: {m['overfitting']:.2f}% {get_overfitting_status(best['overfitting'])}")
        print(f"\n💡 Use for production:")
        print(f"   Model: rl_system/models/{best['name']}/best/best_model.zip")
        print(f"   Config: rl_system/models/{best['name']}/best/config.json")

def main():
    models_dir = Path('rl_system/models')
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Собираем все модели
    models = []
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            metrics = load_model_metrics(model_path)
            if metrics:
                models.append(metrics)
    
    if not models:
        print("❌ No models found with valid configs")
        return
    
    print_comparison_table(models)
    
    print(f"\n📊 Legend:")
    print(f"   PF = Profit Factor (gross_profit / gross_loss)")
    print(f"   Sharpe = Risk-adjusted returns")
    print(f"   MaxDD = Maximum Drawdown")
    print(f"   Overfit = |BEST return - FINAL return|")
    print(f"   ✅ Stable (< 3%), ⚠️  Moderate (3-10%), ❌ Strong (> 10%)")

if __name__ == '__main__':
    main()
