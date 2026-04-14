"""
Скрипт для пересоздания визуализации чекпоинтов для уже обученных моделей
"""

import argparse
import json
from pathlib import Path
import sys

# Добавляем путь к модулям
sys.path.append('rl_system')

from select_best_model import select_best_checkpoint
from plateau_analysis import compute_plateau, compute_live_verdict, load_thresholds_from_config

# Импортируем функции визуализации из train_agent_v2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_checkpoints_comparison(metrics_list: list, model_dir: Path, best_checkpoint_name: str):
    """
    Создает график сравнения всех чекпоинтов
    
    Args:
        metrics_list: Список метрик всех чекпоинтов из select_best_checkpoint
        model_dir: Путь к директории модели
        best_checkpoint_name: Имя выбранного лучшего чекпоинта
    """
    if not metrics_list:
        print("⚠️  No checkpoint metrics available for plotting")
        return
    
    # Сортируем по timestep
    metrics_list = sorted(metrics_list, key=lambda x: x.get('timestep', 0))
    
    # Фильтруем только реальные чекпоинты (исключаем best/final с специальными timestep)
    real_metrics = [m for m in metrics_list if m.get('timestep', 0) < 900000000]
    
    if not real_metrics:
        print("⚠️  No real checkpoint metrics available (only best/final models)")
        return
    
    # Извлекаем данные только из реальных чекпоинтов
    timesteps = [m.get('timestep', 0) for m in real_metrics]
    returns = [m.get('total_return_pct', 0) for m in real_metrics]
    profit_factors = [m.get('profit_factor', 0) for m in real_metrics]
    trades = [m.get('total_trades', 0) for m in real_metrics]
    win_rates = [m.get('win_rate_pct', 0) for m in real_metrics]
    scores = [m.get('score', 0) for m in real_metrics]
    checkpoints = [m.get('checkpoint', '') for m in real_metrics]
    
    # Найдем индекс лучшего чекпоинта среди реальных чекпоинтов
    best_idx = None
    for i, cp in enumerate(checkpoints):
        if best_checkpoint_name in cp:
            best_idx = i
            break
    
    # Если best_checkpoint это best_model.zip, ищем его метрики в полном списке
    if best_idx is None:
        best_checkpoint_metrics = next((m for m in metrics_list if best_checkpoint_name in m.get('checkpoint', '')), None)
        if best_checkpoint_metrics:
            # Находим ближайший чекпоинт по метрикам
            best_return = best_checkpoint_metrics.get('total_return_pct', 0)
            best_pf = best_checkpoint_metrics.get('profit_factor', 0)
            
            # Ищем наиболее похожий чекпоинт
            min_diff = float('inf')
            for i, m in enumerate(real_metrics):
                diff = abs(m.get('total_return_pct', 0) - best_return) + abs(m.get('profit_factor', 0) - best_pf)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Checkpoint Quality Comparison', fontsize=16, fontweight='bold')
    
    # 1. Return %
    ax1 = axes[0, 0]
    ax1.plot(timesteps, returns, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    if best_idx is not None:
        ax1.plot(timesteps[best_idx], returns[best_idx], 'r*', markersize=20, 
                label=f'Best: {returns[best_idx]:.2f}%', zorder=5)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Return %')
    ax1.set_title('Return % vs Training Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Profit Factor
    ax2 = axes[0, 1]
    ax2.plot(timesteps, profit_factors, 'o-', linewidth=2, markersize=6, color='#A23B72')
    if best_idx is not None:
        ax2.plot(timesteps[best_idx], profit_factors[best_idx], 'r*', markersize=20,
                label=f'Best: {profit_factors[best_idx]:.2f}', zorder=5)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Profit Factor')
    ax2.set_title('Profit Factor vs Training Steps')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Number of Trades
    ax3 = axes[1, 0]
    ax3.plot(timesteps, trades, 'o-', linewidth=2, markersize=6, color='#F18F01')
    if best_idx is not None:
        ax3.plot(timesteps[best_idx], trades[best_idx], 'r*', markersize=20,
                label=f'Best: {trades[best_idx]}', zorder=5)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Number of Trades')
    ax3.set_title('Trading Activity vs Training Steps')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Win Rate %
    ax4 = axes[1, 1]
    ax4.plot(timesteps, win_rates, 'o-', linewidth=2, markersize=6, color='#6A994E')
    if best_idx is not None:
        ax4.plot(timesteps[best_idx], win_rates[best_idx], 'r*', markersize=20,
                label=f'Best: {win_rates[best_idx]:.1f}%', zorder=5)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Win Rate %')
    ax4.set_title('Win Rate % vs Training Steps')
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Score (Combined Metric)
    ax5 = axes[2, 0]
    ax5.plot(timesteps, scores, 'o-', linewidth=2, markersize=6, color='#BC4B51')
    if best_idx is not None:
        ax5.plot(timesteps[best_idx], scores[best_idx], 'r*', markersize=20,
                label=f'Best: {scores[best_idx]:.2f}', zorder=5)
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Combined Score')
    ax5.set_title('Combined Score vs Training Steps (Selection Criterion)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Summary Table
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Подготовка данных для таблицы
    if best_idx is not None:
        summary_data = [
            ['Metric', 'Best Checkpoint', 'Final Checkpoint'],
            ['Timestep', f"{timesteps[best_idx]:,}", f"{timesteps[-1]:,}"],
            ['Return %', f"{returns[best_idx]:+.2f}%", f"{returns[-1]:+.2f}%"],
            ['Profit Factor', f"{profit_factors[best_idx]:.2f}", f"{profit_factors[-1]:.2f}"],
            ['Trades', f"{trades[best_idx]}", f"{trades[-1]}"],
            ['Win Rate %', f"{win_rates[best_idx]:.1f}%", f"{win_rates[-1]:.1f}%"],
            ['Score', f"{scores[best_idx]:.2f}", f"{scores[-1]:.2f}"],
        ]
        
        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Выделяем заголовок
        for i in range(3):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
        
        # Выделяем лучшую колонку
        for i in range(1, len(summary_data)):
            table[(i, 1)].set_facecolor('#FFFACD')
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = model_dir / 'checkpoints_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Checkpoint comparison plot saved: {plot_path}")
    plt.close()


def create_checkpoints_markdown(metrics_list: list, model_dir: Path, best_checkpoint_name: str):
    """
    Создает markdown файл с таблицей сравнения чекпоинтов
    
    Args:
        metrics_list: Список метрик всех чекпоинтов
        model_dir: Путь к директории модели
        best_checkpoint_name: Имя выбранного лучшего чекпоинта
    """
    if not metrics_list:
        print("⚠️  No checkpoint metrics available for markdown")
        return
    
    # Сортируем по timestep
    metrics_list = sorted(metrics_list, key=lambda x: x.get('timestep', 0))

    thresholds = load_thresholds_from_config(model_dir)
    plateau = compute_plateau(metrics_list, best_checkpoint_name, thresholds)
    live_verdict = compute_live_verdict(metrics_list, best_checkpoint_name, thresholds, plateau)
    
    md_path = model_dir / 'CHECKPOINTS_COMPARISON.md'
    
    # Загружаем config.json для отображения параметров обучения
    config_path = model_dir / 'config.json'
    config_data = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as cf:
            config_data = json.load(cf)
    
    # Загружаем информацию о min_trades из selected_best_by_metrics.json
    min_trades_info = None
    selector_config_path = model_dir / "selected_best_by_metrics.json"
    if selector_config_path.exists():
        with open(selector_config_path, 'r', encoding='utf-8') as sf:
            selector_data = json.load(sf)
            min_trades_info = {
                'min_trades': selector_data.get('min_trades', 'N/A'),
                'timeframe': selector_data.get('min_trades_info', {}).get('timeframe', 'N/A'),
                'test_days': selector_data.get('min_trades_info', {}).get('test_days', 'N/A'),
                'test_bars': selector_data.get('min_trades_info', {}).get('test_bars', 'N/A')
            }
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Checkpoints Comparison\n\n")
        f.write(f"**Model:** `{model_dir.name}`\n\n")
        
        # Добавляем имя конфиг-файла, если оно есть
        if config_data and config_data.get('config_file'):
            config_file = config_data.get('config_file')
            f.write(f"**Config File:** `{config_file}`\n\n")
        
        f.write(f"**Best Checkpoint:** `{best_checkpoint_name}` ⭐\n\n")
        
        # Добавляем информацию о минимальном количестве сделок
        if min_trades_info:
            f.write(f"**Min Trades Threshold:** {min_trades_info['min_trades']} ")
            f.write(f"(calculated for {min_trades_info['timeframe']} timeframe, ")
            f.write(f"{min_trades_info['test_days']} test days, ")
            f.write(f"{min_trades_info['test_bars']} test bars)\n\n")
        
        f.write("---\n\n")
        
        # Таблица со всеми чекпоинтами
        f.write("## All Checkpoints Performance\n\n")
        f.write("| Step | Episodes | Return % | PF | Trades | Win % | Max DD % | Sharpe | Score | Status |\n")
        f.write("|------|----------|----------|----|----|-------|----------|--------|-------|--------|\n")
        
        for m in metrics_list:
            checkpoint = m.get('checkpoint', 'N/A')
            step = m.get('timestep', 0)
            episodes = m.get('episodes', 0)
            ret = m.get('total_return_pct', 0)
            pf = m.get('profit_factor', 0)
            trades = m.get('total_trades', 0)
            wr = m.get('win_rate_pct', 0)
            dd = m.get('max_drawdown_pct', 0)
            sharpe = m.get('sharpe_ratio', 0)
            score = m.get('score', 0)
            
            # Определяем статус
            is_best = best_checkpoint_name in checkpoint
            status = "⭐ **BEST**" if is_best else ""
            
            # Форматируем строку
            episodes_str = f"{episodes:.1f}" if episodes > 0 else "-"
            
            f.write(f"| {step:,} | {episodes_str} | {ret:+.2f} | {pf:.2f} | {trades} | {wr:.1f} | {dd:.2f} | {sharpe:.2f} | {score:.2f} | {status} |\n")
        
        f.write("\n---\n\n")
        
        # Детальное сравнение Best vs Final
        best_metrics = next((m for m in metrics_list if best_checkpoint_name in m.get('checkpoint', '')), None)
        # Финальный чекпоинт - это последний чекпоинт с реальным timestep (не best/final)
        real_checkpoints = [m for m in metrics_list if m.get('timestep', 0) < 900000000]
        final_metrics = real_checkpoints[-1] if real_checkpoints else metrics_list[-1]
        
        if best_metrics:
            f.write("## Best vs Final Checkpoint\n\n")
            f.write("| Metric | Best Checkpoint ⭐ | Final Checkpoint | Difference |\n")
            f.write("|--------|-------------------|------------------|------------|\n")
            
            metrics_to_compare = [
                ('Timestep', 'timestep', ',d'),
                ('Return %', 'total_return_pct', '+.2f'),
                ('Profit Factor', 'profit_factor', '.2f'),
                ('Trades', 'total_trades', 'd'),
                ('Win Rate %', 'win_rate_pct', '.1f'),
                ('Max Drawdown %', 'max_drawdown_pct', '.2f'),
                ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
                ('Score', 'score', '.2f'),
            ]
            
            for label, key, fmt in metrics_to_compare:
                best_val = best_metrics.get(key, 0)
                final_val = final_metrics.get(key, 0)
                
                if 'd' in fmt:
                    diff = final_val - best_val
                    diff_str = f"{diff:+d}" if diff != 0 else "="
                    best_str = f"{best_val:{fmt}}"
                    final_str = f"{final_val:{fmt}}"
                else:
                    diff = final_val - best_val
                    diff_str = f"{diff:+.2f}" if abs(diff) > 0.01 else "="
                    best_str = f"{best_val:{fmt}}"
                    final_str = f"{final_val:{fmt}}"
                
                f.write(f"| {label} | {best_str} | {final_str} | {diff_str} |\n")
            
            f.write("\n---\n\n")
        
        # Анализ переобучения
        if best_metrics:
            best_return = best_metrics.get('total_return_pct', 0)
            final_return = final_metrics.get('total_return_pct', 0)
            overfitting = best_return - final_return
            
            f.write("## Overfitting Analysis\n\n")
            f.write(f"- **Best Model Return:** {best_return:+.2f}%\n")
            f.write(f"- **Final Model Return:** {final_return:+.2f}%\n")
            f.write(f"- **Difference:** {overfitting:.2f}%\n\n")
            
            if abs(overfitting) < 3:
                f.write("✅ **Status:** Stable (minimal overfitting)\n\n")
            elif abs(overfitting) < 10:
                f.write("⚠️  **Status:** Moderate overfitting\n\n")
            else:
                f.write("❌ **Status:** Strong overfitting - consider reducing timesteps or adjusting hyperparameters\n\n")

        # Анализ плато / стабильности + вердикт пригодности
        f.write("## Plateau / Stability Analysis\n\n")
        f.write("This section checks whether performance forms a stable region (plateau) rather than an isolated spike.\n\n")
        if isinstance(plateau, dict) and plateau.get("best_score") is not None:
            f.write(f"- **Plateau criterion:** score ≥ best_score - eps\n")
            f.write(f"- **best_score:** {plateau.get('best_score', 0):+.2f}\n")
            f.write(f"- **eps used:** {plateau.get('eps_used', 0):.2f} (abs={thresholds.score_eps_abs:.2f}, rel={thresholds.score_eps_rel:.2f})\n")
            f.write(f"- **min plateau length:** {thresholds.plateau_min_len} consecutive checkpoints\n")
            f.write(f"- **valid checkpoints gate:** trades ≥ {thresholds.min_trades} AND episodes ≥ 2\n\n")

        plateau_len = plateau.get("len", 0) if isinstance(plateau, dict) else 0
        if isinstance(plateau, dict) and plateau.get("found", False):
            f.write(f"✅ **Plateau detected:** {plateau_len} checkpoints ({plateau.get('start_step', 'N/A'):,} → {plateau.get('end_step', 'N/A'):,})\n\n")
            f.write("| Step | Return % | PF | Trades | Max DD % | Score |\n")
            f.write("|------|----------|----|--------|----------|-------|\n")
            for p in plateau.get("steps", [])[:25]:
                f.write(
                    f"| {int(p.get('timestep', 0)):,} | {float(p.get('return_pct', 0)):+.2f} | {float(p.get('pf', 0)):.2f} | {int(p.get('trades', 0))} | {float(p.get('dd', 0)):.2f} | {float(p.get('score', 0)):+.2f} |\n"
                )
            if len(plateau.get("steps", [])) > 25:
                f.write(f"\n*Plateau list truncated to first 25 checkpoints (total: {len(plateau.get('steps', []))}).*\n\n")
            else:
                f.write("\n")
        else:
            min_len = plateau.get("min_len", thresholds.plateau_min_len) if isinstance(plateau, dict) else thresholds.plateau_min_len
            reason = plateau.get("reason", "plateau_not_found") if isinstance(plateau, dict) else "plateau_not_found"
            f.write(f"❌ **No stable plateau:** longest region below threshold is {plateau_len} checkpoints (min {min_len}).\n")
            f.write(f"- **Reason:** {reason}\n\n")

        f.write("## Live Readiness Verdict\n\n")
        verdict_status = live_verdict.get('status', 'FAIL')
        if verdict_status == 'PASS':
            f.write("✅ **PASS:** model looks stable enough to proceed to paper/forward testing.\n\n")
        else:
            f.write("❌ **FAIL:** do NOT treat this model as trade-ready based on checkpoint backtests.\n\n")

        f.write("**Decision thresholds (defaults or config.json overrides):**\n\n")
        th = live_verdict.get('thresholds', {})
        f.write(f"- min_trades: {th.get('min_trades')}\n")
        f.write(f"- pf_min: {th.get('pf_min')}\n")
        f.write(f"- dd_max: {th.get('dd_max')}\n")
        f.write(f"- min_return_pct: {th.get('min_return_pct')}\n")
        f.write(f"- plateau_min_len: {th.get('plateau_min_len')}\n")
        f.write(f"- tail_len: {th.get('tail_len')}\n")
        f.write(f"- tail_min_profitable_ratio: {th.get('tail_min_profitable_ratio')}\n\n")

        reasons = live_verdict.get('reasons', [])
        if reasons:
            f.write("**Reasons:**\n\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # Сравнение со специальными моделями (best_model.zip, final_model.zip)
        special_models = [m for m in metrics_list if m.get('timestep', 0) >= 900000000]
        if special_models:
            f.write("## Comparison with Special Models\n\n")
            f.write("*These models were NOT included in the selection process. Shown for reference only.*\n\n")
            f.write("| Model | Description | Return % | PF | Trades | Win % | Score |\n")
            f.write("|-------|-------------|----------|----|----|-------|-------|\n")
            
            for sm in special_models:
                model_name = sm.get('checkpoint', 'N/A')
                ret = sm.get('total_return_pct', 0)
                pf = sm.get('profit_factor', 0)
                trades = sm.get('total_trades', 0)
                wr = sm.get('win_rate_pct', 0)
                score = sm.get('score', 0)
                
                if 'best' in model_name.lower():
                    desc = "Saved by Stable-Baselines3 during training (best validation reward)"
                elif 'final' in model_name.lower():
                    desc = "Last checkpoint at end of training"
                else:
                    desc = "Special model"
                
                f.write(f"| {model_name} | {desc} | {ret:+.2f} | {pf:.2f} | {trades} | {wr:.1f} | {score:.2f} |\n")
            
            # Сравнение лучшего регулярного чекпоинта с best_model.zip
            if best_metrics:
                best_model = next((m for m in special_models if 'best' in m.get('checkpoint', '').lower()), None)
                if best_model:
                    f.write("\n### Selected Checkpoint vs Stable-Baselines3 Best Model\n\n")
                    best_return = best_metrics.get('total_return_pct', 0)
                    sb3_return = best_model.get('total_return_pct', 0)
                    diff = best_return - sb3_return
                    
                    f.write(f"- **Selected Checkpoint Return:** {best_return:+.2f}%\n")
                    f.write(f"- **SB3 Best Model Return:** {sb3_return:+.2f}%\n")
                    f.write(f"- **Difference:** {diff:+.2f}%\n\n")
                    
                    if diff > 0:
                        f.write("✅ **Selected checkpoint outperforms SB3's best model on trading metrics**\n\n")
                    elif diff < -1:
                        f.write("⚠️  **SB3's best model outperforms selected checkpoint**\n\n")
                    else:
                        f.write("➡️  **Similar performance**\n\n")
            
            f.write("---\n\n")
        
        f.write("*Generated automatically after training completion*\n")
    
    print(f"📝 Checkpoint comparison markdown saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate checkpoint visualization for trained model")
    parser.add_argument("--model-dir", type=str, required=True, 
                       help="Path to model directory (e.g. rl_system/models/...)")
    parser.add_argument("--min-trades", type=int, default=30,
                       help="Minimum number of trades to consider model valid")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"🔄 Regenerating visualization for: {model_dir.name}")
    
    try:
        # Запускаем селектор для получения метрик
        print(f"\n🔍 Running checkpoint selector...")
        metrics_list = select_best_checkpoint(model_dir, min_trades=args.min_trades)
        
        # Загружаем информацию о выбранном лучшем чекпоинте
        selector_config_path = model_dir / "selected_best_by_metrics.json"
        if selector_config_path.exists():
            with open(selector_config_path, 'r') as f:
                best_selection = json.load(f)
                # Поддерживаем старый и новый формат
                if "best_checkpoint" in best_selection:
                    best_checkpoint_name = best_selection["best_checkpoint"].get('checkpoint', '')
                else:
                    # Старый формат
                    best_checkpoint_name = best_selection.get('checkpoint', '')
            
            print("✅ Selector finished")
            
            # Создаем визуализацию и markdown
            if metrics_list:
                print(f"\n📊 Creating checkpoint comparison visualizations...")
                plot_checkpoints_comparison(metrics_list, model_dir, best_checkpoint_name)
                create_checkpoints_markdown(metrics_list, model_dir, best_checkpoint_name)
                print("✅ Visualizations created successfully!")
            else:
                print("⚠️  No metrics available for visualization")
        else:
            print(f"❌ selected_best_by_metrics.json not found")
        
    except Exception as e:
        print(f"❌ Failed to regenerate visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
