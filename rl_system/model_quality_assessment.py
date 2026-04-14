"""
Модуль для комплексной оценки качества RL-моделей для крипто-трейдинга.

Оценивает модель по 4 уровням:
1. Critical Checks - обязательные базовые проверки
2. Risk Assessment - оценка риск-метрик
3. Stability Analysis - анализ устойчивости результатов
4. Learning Quality - качество процесса обучения

Результат: READY FOR LIVE / USE WITH CAUTION / PAPER TRADING ONLY / NOT READY
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json


@dataclass
class QualityThresholds:
    """Пороги для оценки качества модели"""
    # Level 1: Critical
    min_final_return: float = 0.0
    min_final_pf: float = 1.0
    max_dd_critical: float = 30.0
    min_episodes_critical: float = 5.0
    
    # Level 2: Risk
    max_dd_good: float = 25.0
    min_pf_good: float = 1.3
    min_sharpe_good: float = 0.5
    min_return_dd_ratio_good: float = 0.5
    
    min_pf_acceptable: float = 1.0
    min_sharpe_acceptable: float = 0.3
    min_return_dd_ratio_acceptable: float = 0.2
    
    # Level 3: Stability
    min_profitable_zone_pct: float = 15.0  # % от timesteps
    min_profitable_zone_caution: float = 10.0
    
    max_overfitting_good: float = 30.0  # % разницы best-final
    max_overfitting_acceptable: float = 40.0
    
    min_final_stability_profitable_pct: float = 50.0  # % прибыльных в последних 20%
    max_final_stability_std: float = 10.0
    
    min_consistency_pct: float = 50.0  # % прибыльных в последних 30%
    
    # Level 4: Learning
    min_progression: float = 0.0  # Конец должен быть лучше начала
    max_collapse_return: float = -30.0  # Допустимый минимум в последних 25%
    excellent_top_return: float = 10.0
    good_top_return: float = 5.0


def _safe_float(val, default=0.0) -> float:
    """Безопасное преобразование в float"""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    """Безопасное преобразование в int"""
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _is_regular_checkpoint(m: dict) -> bool:
    """Проверка, что чекпоинт не best/final"""
    return _safe_int(m.get("timestep", 0)) < 900000000


def calculate_profitable_zone_coverage(metrics_list: List[dict], total_timesteps: int) -> Dict:
    """
    Рассчитывает процент времени обучения с положительным return.
    
    Returns:
        dict: {
            'coverage_pct': float,  # % timesteps с return > 0
            'profitable_steps': int,
            'total_steps': int,
            'zones': List[dict]  # Непрерывные зоны прибыльности
        }
    """
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    regular = sorted(regular, key=lambda x: _safe_int(x.get("timestep", 0)))
    
    if not regular:
        return {'coverage_pct': 0.0, 'profitable_steps': 0, 'total_steps': 0, 'zones': []}
    
    # Находим непрерывные зоны прибыльности
    zones = []
    current_zone = None
    
    for m in regular:
        ret = _safe_float(m.get("total_return_pct", 0.0))
        step = _safe_int(m.get("timestep", 0))
        
        if ret > 0:
            if current_zone is None:
                current_zone = {
                    'start_step': step,
                    'end_step': step,
                    'returns': [ret],
                    'count': 1
                }
            else:
                current_zone['end_step'] = step
                current_zone['returns'].append(ret)
                current_zone['count'] += 1
        else:
            if current_zone is not None:
                current_zone['duration'] = current_zone['end_step'] - current_zone['start_step']
                current_zone['mean_return'] = sum(current_zone['returns']) / len(current_zone['returns'])
                zones.append(current_zone)
                current_zone = None
    
    # Не забываем последнюю зону
    if current_zone is not None:
        current_zone['duration'] = current_zone['end_step'] - current_zone['start_step']
        current_zone['mean_return'] = sum(current_zone['returns']) / len(current_zone['returns'])
        zones.append(current_zone)
    
    # Суммируем длительность всех зон
    total_profitable_steps = sum(z['duration'] for z in zones)
    coverage_pct = (total_profitable_steps / total_timesteps * 100) if total_timesteps > 0 else 0.0
    
    return {
        'coverage_pct': coverage_pct,
        'profitable_steps': total_profitable_steps,
        'total_steps': total_timesteps,
        'zones': sorted(zones, key=lambda z: z['duration'], reverse=True)  # От большей к меньшей
    }


def analyze_final_stability(metrics_list: List[dict], total_timesteps: int, tail_pct: float = 0.20) -> Dict:
    """
    Анализирует стабильность в последних N% обучения.
    
    Args:
        tail_pct: Процент от конца обучения для анализа (по умолчанию 20%)
    
    Returns:
        dict: {
            'profitable_pct': float,
            'mean_return': float,
            'std_return': float,
            'total_checkpoints': int,
            'profitable_checkpoints': int
        }
    """
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    
    if not regular:
        return {'profitable_pct': 0.0, 'mean_return': 0.0, 'std_return': 0.0, 
                'total_checkpoints': 0, 'profitable_checkpoints': 0}
    
    cutoff_step = total_timesteps * (1 - tail_pct)
    tail_checkpoints = [m for m in regular if _safe_int(m.get("timestep", 0)) >= cutoff_step]
    
    if not tail_checkpoints:
        return {'profitable_pct': 0.0, 'mean_return': 0.0, 'std_return': 0.0,
                'total_checkpoints': 0, 'profitable_checkpoints': 0}
    
    returns = [_safe_float(m.get("total_return_pct", 0.0)) for m in tail_checkpoints]
    profitable = [r for r in returns if r > 0]
    
    mean_return = sum(returns) / len(returns) if returns else 0.0
    
    # Вычисляем стандартное отклонение
    if len(returns) > 1:
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5
    else:
        std_return = 0.0
    
    return {
        'profitable_pct': (len(profitable) / len(returns) * 100) if returns else 0.0,
        'mean_return': mean_return,
        'std_return': std_return,
        'total_checkpoints': len(tail_checkpoints),
        'profitable_checkpoints': len(profitable),
        'cutoff_step': int(cutoff_step)
    }


def detect_learning_progression(metrics_list: List[dict], total_timesteps: int) -> Dict:
    """
    Определяет прогрессию обучения: улучшается ли модель со временем.
    
    Returns:
        dict: {
            'first_segment_mean': float,
            'last_segment_mean': float,
            'improvement': float,
            'has_progression': bool
        }
    """
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    
    if len(regular) < 10:
        return {'first_segment_mean': 0.0, 'last_segment_mean': 0.0, 
                'improvement': 0.0, 'has_progression': False}
    
    # Берем первые 12.5% и последние 12.5% (100k steps из 800k)
    segment_size = total_timesteps * 0.125
    
    first_segment = [m for m in regular if _safe_int(m.get("timestep", 0)) <= segment_size]
    last_segment = [m for m in regular if _safe_int(m.get("timestep", 0)) >= (total_timesteps - segment_size)]
    
    first_returns = [_safe_float(m.get("total_return_pct", 0.0)) for m in first_segment]
    last_returns = [_safe_float(m.get("total_return_pct", 0.0)) for m in last_segment]
    
    first_mean = sum(first_returns) / len(first_returns) if first_returns else 0.0
    last_mean = sum(last_returns) / len(last_returns) if last_returns else 0.0
    
    improvement = last_mean - first_mean
    
    return {
        'first_segment_mean': first_mean,
        'last_segment_mean': last_mean,
        'improvement': improvement,
        'has_progression': improvement > 0,
        'segment_size_steps': int(segment_size)
    }


def check_catastrophic_collapse(metrics_list: List[dict], total_timesteps: int, 
                                 tail_pct: float = 0.25) -> Dict:
    """
    Проверяет наличие катастрофического коллапса в финале обучения.
    
    Returns:
        dict: {
            'worst_return': float,
            'has_collapse': bool,
            'worst_checkpoint': str
        }
    """
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    cutoff_step = total_timesteps * (1 - tail_pct)
    
    tail_checkpoints = [m for m in regular if _safe_int(m.get("timestep", 0)) >= cutoff_step]
    
    if not tail_checkpoints:
        return {'worst_return': 0.0, 'has_collapse': False, 'worst_checkpoint': 'N/A'}
    
    worst = min(tail_checkpoints, key=lambda m: _safe_float(m.get("total_return_pct", 0.0)))
    worst_return = _safe_float(worst.get("total_return_pct", 0.0))
    
    return {
        'worst_return': worst_return,
        'has_collapse': worst_return < -30.0,
        'worst_checkpoint': worst.get('checkpoint', 'N/A'),
        'worst_step': _safe_int(worst.get('timestep', 0))
    }


def calculate_overfitting_score(best_return: float, final_return: float) -> Dict:
    """
    Рассчитывает степень переобучения.
    
    Returns:
        dict: {
            'difference_pct': float,
            'overfitting_level': str  # 'GOOD', 'ACCEPTABLE', 'WARNING', 'SEVERE'
        }
    """
    diff = best_return - final_return
    diff_pct = (diff / abs(best_return) * 100) if best_return != 0 else 0.0
    
    if diff_pct < 30:
        level = 'GOOD'
    elif diff_pct < 40:
        level = 'ACCEPTABLE'
    elif diff_pct < 60:
        level = 'WARNING'
    else:
        level = 'SEVERE'
    
    return {
        'difference': diff,
        'difference_pct': diff_pct,
        'overfitting_level': level
    }


def analyze_top_performers(metrics_list: List[dict], top_n: int = 10) -> Dict:
    """
    Анализирует качество топ-N чекпоинтов.
    
    Returns:
        dict: {
            'mean_return': float,
            'quality_rating': str  # 'EXCELLENT', 'GOOD', 'WEAK'
        }
    """
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    
    if not regular:
        return {'mean_return': 0.0, 'quality_rating': 'WEAK'}
    
    # Сортируем по return и берем top N
    sorted_by_return = sorted(regular, key=lambda m: _safe_float(m.get("total_return_pct", 0.0)), reverse=True)
    top_performers = sorted_by_return[:top_n]
    
    returns = [_safe_float(m.get("total_return_pct", 0.0)) for m in top_performers]
    mean_return = sum(returns) / len(returns) if returns else 0.0
    
    if mean_return > 10.0:
        rating = 'EXCELLENT'
    elif mean_return > 5.0:
        rating = 'GOOD'
    else:
        rating = 'WEAK'
    
    return {
        'mean_return': mean_return,
        'quality_rating': rating,
        'top_checkpoints': [
            {
                'checkpoint': m.get('checkpoint', ''),
                'step': _safe_int(m.get('timestep', 0)),
                'return': _safe_float(m.get('total_return_pct', 0.0)),
                'pf': _safe_float(m.get('profit_factor', 0.0))
            }
            for m in top_performers
        ]
    }


def assess_model_quality(metrics_list: List[dict], model_dir: Path, 
                         best_checkpoint_name: str, 
                         thresholds: QualityThresholds = None) -> Dict:
    """
    Комплексная оценка качества модели по всем 4 уровням.
    
    Returns:
        dict: Полный отчет с вердиктом и рекомендациями
    """
    if thresholds is None:
        thresholds = QualityThresholds()
    
    # Загружаем config.json для получения параметров
    config_path = model_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    total_timesteps = config.get('total_timesteps', 800000)
    min_trades = config.get('min_trades', 30)  # Из plateau_analysis или config
    
    # Находим final и best модели
    # Final model может быть: final_model.zip, best_model.zip, или последний чекпоинт по timestep
    final_model = next((m for m in metrics_list if 'final_model' in m.get('checkpoint', '')), None)
    
    # Если нет final_model.zip (старые модели), используем best_model.zip как финальную
    if not final_model:
        final_model = next((m for m in metrics_list if 'best_model.zip' in m.get('checkpoint', '')), None)
    
    # Если и best_model.zip нет, берем чекпоинт с максимальным timestep
    if not final_model:
        regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
        if regular:
            final_model = max(regular, key=lambda m: _safe_int(m.get('timestep', 0)))
    
    # Находим best чекпоинт по имени
    best_model = next((m for m in metrics_list if best_checkpoint_name in m.get('checkpoint', '')), None)
    
    if not final_model or not best_model:
        return {'error': f'Final or best model not found in metrics. Looking for: {best_checkpoint_name}'}
    
    # Извлекаем метрики final модели
    final_return = _safe_float(final_model.get('total_return_pct', 0.0))
    final_pf = _safe_float(final_model.get('profit_factor', 0.0))
    final_dd = _safe_float(final_model.get('max_drawdown_pct', 0.0))
    final_sharpe = _safe_float(final_model.get('sharpe_ratio', 0.0))
    final_trades = _safe_int(final_model.get('total_trades', 0))
    final_episodes = _safe_float(final_model.get('episodes', 0.0))
    
    best_return = _safe_float(best_model.get('total_return_pct', 0.0))
    best_pf = _safe_float(best_model.get('profit_factor', 0.0))
    best_dd = _safe_float(best_model.get('max_drawdown_pct', 0.0))
    
    # === LEVEL 1: Critical Checks ===
    level1_checks = {
        'final_profitable': final_return > thresholds.min_final_return and final_pf > thresholds.min_final_pf,
        'dd_acceptable': final_dd <= thresholds.max_dd_critical,
        'min_trades_met': final_trades >= min_trades,
        'min_episodes_met': final_episodes >= thresholds.min_episodes_critical
    }
    
    level1_pass = all(level1_checks.values())
    level1_score = sum(level1_checks.values())
    
    # === LEVEL 2: Risk Assessment ===
    return_dd_ratio = final_return / final_dd if final_dd > 0 else 0.0
    
    level2_results = {
        'max_dd': {
            'value': final_dd,
            'status': 'PASS' if final_dd <= thresholds.max_dd_good else 'CAUTION' if final_dd <= thresholds.max_dd_critical else 'FAIL'
        },
        'profit_factor': {
            'value': final_pf,
            'status': 'PASS' if final_pf >= thresholds.min_pf_good else 'CAUTION' if final_pf >= thresholds.min_pf_acceptable else 'FAIL'
        },
        'sharpe_ratio': {
            'value': final_sharpe,
            'status': 'PASS' if final_sharpe >= thresholds.min_sharpe_good else 'CAUTION' if final_sharpe >= thresholds.min_sharpe_acceptable else 'FAIL'
        },
        'return_dd_ratio': {
            'value': return_dd_ratio,
            'status': 'PASS' if return_dd_ratio >= thresholds.min_return_dd_ratio_good else 'CAUTION' if return_dd_ratio >= thresholds.min_return_dd_ratio_acceptable else 'FAIL'
        }
    }
    
    level2_pass_count = sum(1 for r in level2_results.values() if r['status'] == 'PASS')
    level2_score_pct = (level2_pass_count / len(level2_results)) * 100
    
    # === LEVEL 3: Stability Analysis ===
    profitable_zone = calculate_profitable_zone_coverage(metrics_list, total_timesteps)
    final_stability = analyze_final_stability(metrics_list, total_timesteps, tail_pct=0.20)
    overfitting = calculate_overfitting_score(best_return, final_return)
    
    # Консистентность в последних 30%
    consistency_30 = analyze_final_stability(metrics_list, total_timesteps, tail_pct=0.30)
    
    level3_results = {
        'profitable_zone': {
            'coverage_pct': profitable_zone['coverage_pct'],
            'status': 'PASS' if profitable_zone['coverage_pct'] >= thresholds.min_profitable_zone_pct else 'CAUTION' if profitable_zone['coverage_pct'] >= thresholds.min_profitable_zone_caution else 'FAIL',
            'zones': profitable_zone['zones']
        },
        'final_stability': {
            'mean_return': final_stability['mean_return'],
            'std_return': final_stability['std_return'],
            'profitable_pct': final_stability['profitable_pct'],
            'status': 'PASS' if (final_stability['mean_return'] > 0 and 
                                final_stability['std_return'] < thresholds.max_final_stability_std) else 'FAIL'
        },
        'overfitting': {
            'difference_pct': overfitting['difference_pct'],
            'level': overfitting['overfitting_level'],
            'status': 'PASS' if overfitting['difference_pct'] < thresholds.max_overfitting_good else 'CAUTION' if overfitting['difference_pct'] < thresholds.max_overfitting_acceptable else 'FAIL'
        },
        'consistency_30': {
            'profitable_pct': consistency_30['profitable_pct'],
            'status': 'PASS' if consistency_30['profitable_pct'] >= thresholds.min_consistency_pct else 'FAIL'
        }
    }
    
    level3_pass_count = sum(1 for r in level3_results.values() if r['status'] == 'PASS')
    level3_score_pct = (level3_pass_count / len(level3_results)) * 100
    
    # === LEVEL 4: Learning Quality ===
    progression = detect_learning_progression(metrics_list, total_timesteps)
    collapse_check = check_catastrophic_collapse(metrics_list, total_timesteps, tail_pct=0.25)
    top_performers = analyze_top_performers(metrics_list, top_n=10)
    
    level4_results = {
        'progression': {
            'improvement': progression['improvement'],
            'has_progression': progression['has_progression'],
            'status': 'PASS' if progression['has_progression'] else 'FAIL'
        },
        'no_collapse': {
            'worst_return': collapse_check['worst_return'],
            'has_collapse': collapse_check['has_collapse'],
            'status': 'PASS' if not collapse_check['has_collapse'] else 'FAIL'
        },
        'top_performers': {
            'mean_return': top_performers['mean_return'],
            'rating': top_performers['quality_rating'],
            'status': 'PASS'  # Всегда PASS, это бонусная метрика
        }
    }
    
    level4_pass_count = sum(1 for r in level4_results.values() if r['status'] == 'PASS')
    level4_score_pct = (level4_pass_count / len(level4_results)) * 100
    
    # === FINAL VERDICT ===
    total_criteria = level1_score + level2_pass_count + level3_pass_count + level4_pass_count
    max_criteria = len(level1_checks) + len(level2_results) + len(level3_results) + len(level4_results)
    overall_score_pct = (total_criteria / max_criteria) * 100
    
    # Определяем итоговый статус
    if not level1_pass:
        verdict = 'NOT READY'
        confidence = 0
    elif level2_score_pct >= 75 and level3_score_pct >= 75:
        verdict = 'READY FOR LIVE'
        confidence = 85
    elif level2_score_pct >= 50 and level3_score_pct >= 50:
        verdict = 'USE WITH CAUTION'
        confidence = 65
    else:
        verdict = 'PAPER TRADING ONLY'
        confidence = 40
    
    # Формируем рекомендации
    recommendations = _generate_recommendations(
        verdict, level2_results, level3_results, 
        best_model, final_return, final_dd, best_return
    )
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'overall_score_pct': overall_score_pct,
        'total_criteria_passed': total_criteria,
        'max_criteria': max_criteria,
        
        'level1': {
            'pass': level1_pass,
            'score': level1_score,
            'max_score': len(level1_checks),
            'checks': level1_checks,
            'details': {
                'final_return': final_return,
                'final_pf': final_pf,
                'final_dd': final_dd,
                'final_trades': final_trades,
                'final_episodes': final_episodes,
                'min_trades_threshold': min_trades
            }
        },
        
        'level2': {
            'score_pct': level2_score_pct,
            'pass_count': level2_pass_count,
            'max_count': len(level2_results),
            'results': level2_results
        },
        
        'level3': {
            'score_pct': level3_score_pct,
            'pass_count': level3_pass_count,
            'max_count': len(level3_results),
            'results': level3_results
        },
        
        'level4': {
            'score_pct': level4_score_pct,
            'pass_count': level4_pass_count,
            'max_count': len(level4_results),
            'results': level4_results,
            'top_performers': top_performers
        },
        
        'recommendations': recommendations,
        
        'best_checkpoint': {
            'name': best_checkpoint_name,
            'return': best_return,
            'pf': best_pf,
            'dd': best_dd
        }
    }


def _generate_recommendations(verdict: str, level2: Dict, level3: Dict, 
                              best_model: Dict, final_return: float, 
                              final_dd: float, best_return: float) -> Dict:
    """Генерирует рекомендации на основе результатов оценки"""
    
    recommendations = {
        'for_live_trading': [],
        'risk_management': [],
        'monitoring_plan': []
    }
    
    # Рекомендации по выбору чекпоинта
    best_checkpoint = best_model.get('checkpoint', 'N/A').replace('rl_model_', '').replace('_steps.zip', '')
    best_pf = _safe_float(best_model.get('profit_factor', 0.0))
    
    if verdict == 'READY FOR LIVE' or verdict == 'USE WITH CAUTION':
        recommendations['for_live_trading'].append(
            f"✅ Use checkpoint {best_checkpoint} (+{best_return:.2f}%, PF {best_pf:.2f})"
        )
    
    # Рекомендации по размеру позиции
    if level2['profit_factor']['status'] == 'CAUTION':
        recommendations['for_live_trading'].append(
            "⚠️ Reduce position size to 0.10-0.15 (instead of 0.20) due to borderline PF"
        )
    
    if final_dd > 20:
        recommendations['for_live_trading'].append(
            "⚠️ Set stop-loss at 2% (current: 1%) for extra safety"
        )
    
    if verdict == 'USE WITH CAUTION':
        recommendations['for_live_trading'].append(
            "⚠️ Start with 50% capital allocation, increase after 1 month of positive results"
        )
    
    recommendations['for_live_trading'].append(
        "✅ Monitor return/drawdown in real-time, stop if DD exceeds 20%"
    )
    
    # Risk Management
    expected_return_per_trade = best_return / _safe_int(best_model.get('total_trades', 1))
    recommendations['risk_management'].extend([
        f"Expected return per trade: ~{expected_return_per_trade:.2f}%",
        f"Expected max drawdown: {_safe_float(best_model.get('max_drawdown_pct', 0.0)):.2f}%",
        "Recommended capital: $10,000+ (to handle DD without emotional stress)",
        "Max risk per trade: $100-150 (1.0-1.5%)"
    ])
    
    # Monitoring Plan
    recommendations['monitoring_plan'].extend([
        "✅ Review performance weekly for first month"
    ])
    
    if level2['profit_factor']['value'] < 1.5:
        recommendations['monitoring_plan'].append(
            "⚠️ If PF drops below 1.0 after 30 trades → pause and retrain"
        )
    
    recommendations['monitoring_plan'].append(
        "✅ If return <0% after 50 trades → switch to paper trading"
    )
    
    return recommendations


def format_quality_assessment_markdown(assessment: Dict) -> str:
    """
    Форматирует результаты оценки в markdown для добавления в CHECKPOINTS_COMPARISON.md
    """
    md = "\n\n---\n\n"
    md += "## 🎯 Model Quality Assessment (Production Readiness)\n\n"
    
    # Level 1
    md += "### Level 1: Critical Checks ✅/❌\n\n"
    level1 = assessment['level1']
    checks = level1['checks']
    details = level1['details']
    
    check_final = "✅" if checks['final_profitable'] else "❌"
    check_dd = "✅" if checks['dd_acceptable'] else "❌"
    check_trades = "✅" if checks['min_trades_met'] else "❌"
    check_episodes = "✅" if checks['min_episodes_met'] else "❌"
    
    md += f"- [{check_final}] Final model profitable: {details['final_return']:+.2f}% (PF {details['final_pf']:.2f})\n"
    md += f"- [{check_dd}] Max Drawdown acceptable: {details['final_dd']:.2f}% ≤ 30%\n"
    md += f"- [{check_trades}] Minimum trades met: {details['final_trades']} ≥ {details['min_trades_threshold']}\n"
    md += f"- [{check_episodes}] Minimum episodes: {details['final_episodes']:.1f} ≥ 5\n\n"
    md += f"**Result:** {'✅ PASS' if level1['pass'] else '❌ FAIL'} ({level1['score']}/{level1['max_score']})\n\n"
    md += "---\n\n"
    
    # Level 2
    md += "### Level 2: Risk Assessment 📊\n\n"
    md += "| Metric | Value | Threshold | Status |\n"
    md += "|--------|-------|-----------|--------|\n"
    
    level2 = assessment['level2']['results']
    
    for metric_name, metric_data in level2.items():
        name_display = metric_name.replace('_', ' ').title()
        value = metric_data['value']
        status = metric_data['status']
        
        if metric_name == 'max_dd':
            threshold = "≤25%"
            status_icon = "✅" if status == "PASS" else "⚠️" if status == "CAUTION" else "❌"
        elif metric_name == 'profit_factor':
            threshold = "≥1.3"
            status_icon = "✅" if status == "PASS" else "⚠️" if status == "CAUTION" else "❌"
        elif metric_name == 'sharpe_ratio':
            threshold = "≥0.5"
            status_icon = "✅" if status == "PASS" else "⚠️" if status == "CAUTION" else "❌"
        elif metric_name == 'return_dd_ratio':
            threshold = "≥0.5"
            status_icon = "✅" if status == "PASS" else "⚠️" if status == "CAUTION" else "❌"
        
        md += f"| {name_display} | {value:.2f} | {threshold} | {status_icon} {status} |\n"
    
    level2_score = assessment['level2']
    md += f"\n**Score:** {level2_score['pass_count']}/{level2_score['max_count']} PASS = **{level2_score['score_pct']:.0f}%**\n\n"
    md += "---\n\n"
    
    # Level 3
    md += "### Level 3: Stability Analysis 🔄\n\n"
    level3 = assessment['level3']['results']
    
    md += "**3.1 Profitable Zone Coverage:**\n"
    pz = level3['profitable_zone']
    status_icon = "✅" if pz['status'] == "PASS" else "⚠️" if pz['status'] == "CAUTION" else "❌"
    md += f"- Coverage: **{pz['coverage_pct']:.2f}%** of training {status_icon} {pz['status']}\n"
    
    if pz['zones']:
        top_zone = pz['zones'][0]
        md += f"- Main profitable zone: {top_zone['start_step']:,}-{top_zone['end_step']:,} steps "
        md += f"({top_zone['duration']/1000:.0f}k duration, mean return: +{top_zone['mean_return']:.2f}%)\n"
    
    md += "\n**3.2 Final Checkpoint Stability (last 20%):**\n"
    fs = level3['final_stability']
    status_icon = "✅" if fs['status'] == "PASS" else "❌"
    md += f"- Profitable checkpoints: {fs['profitable_pct']:.1f}% {status_icon}\n"
    md += f"- Mean return: {fs['mean_return']:+.2f}% {status_icon}\n"
    md += f"- Std return: {fs['std_return']:.2f}% {status_icon}\n"
    
    md += "\n**3.3 Overfitting Assessment:**\n"
    of = level3['overfitting']
    best = assessment['best_checkpoint']
    final_ret = assessment['level1']['details']['final_return']
    status_icon = "✅" if of['status'] == "PASS" else "⚠️" if of['status'] == "CAUTION" else "❌"
    md += f"- Best checkpoint: +{best['return']:.2f}%\n"
    md += f"- Final checkpoint: +{final_ret:.2f}%\n"
    md += f"- Difference: {of['difference_pct']:.2f}%\n"
    md += f"- **Status:** {status_icon} {of['level']} (<30% is good)\n"
    
    md += "\n**3.4 Consistency (last 30%):**\n"
    cons = level3['consistency_30']
    status_icon = "✅" if cons['status'] == "PASS" else "❌"
    md += f"- Profitable checkpoints: {cons['profitable_pct']:.1f}% {status_icon} {cons['status']}\n"
    
    level3_score = assessment['level3']
    md += f"\n**Score:** {level3_score['pass_count']}/{level3_score['max_count']} criteria PASS = **{level3_score['score_pct']:.0f}%**\n\n"
    md += "---\n\n"
    
    # Level 4
    md += "### Level 4: Learning Quality 📈\n\n"
    level4 = assessment['level4']['results']
    
    md += "**4.1 Progression Analysis:**\n"
    prog = level4['progression']
    status_icon = "✅" if prog['status'] == "PASS" else "❌"
    md += f"- Improvement from start to end: {prog['improvement']:+.2f}% {status_icon} {prog['status']}\n"
    
    md += "\n**4.2 No Catastrophic Collapse:**\n"
    collapse = level4['no_collapse']
    status_icon = "✅" if collapse['status'] == "PASS" else "❌"
    md += f"- Worst checkpoint in final 25%: {collapse['worst_return']:.2f}%\n"
    md += f"- **Status:** {status_icon} {collapse['status']} (no collapse = return > -30%)\n"
    
    md += "\n**4.3 Top Performers Quality:**\n"
    top = level4['top_performers']
    md += f"- Top 10 checkpoints mean return: {top['mean_return']:+.2f}%\n"
    
    rating_icons = {'EXCELLENT': '⭐⭐⭐', 'GOOD': '⭐⭐', 'WEAK': '⭐'}
    md += f"- **Rating:** {rating_icons.get(top['rating'], '⭐')} {top['rating']}\n"
    
    level4_score = assessment['level4']
    md += f"\n**Score:** {level4_score['pass_count']}/{level4_score['max_count']} criteria PASS = **{level4_score['score_pct']:.0f}%**\n\n"
    md += "---\n\n"
    
    # Final Verdict
    md += "## 🏆 FINAL VERDICT\n\n"
    
    verdict_icons = {
        'READY FOR LIVE': '✅',
        'USE WITH CAUTION': '⚠️',
        'PAPER TRADING ONLY': '⏸️',
        'NOT READY': '❌'
    }
    
    verdict = assessment['verdict']
    icon = verdict_icons.get(verdict, '❓')
    
    md += f"**Status:** {icon} **{verdict}**\n\n"
    md += "**Overall Score:**\n"
    md += f"- Level 1 (Critical): {'✅' if assessment['level1']['pass'] else '❌'} {assessment['level1']['score']}/{assessment['level1']['max_score']}\n"
    md += f"- Level 2 (Risk): {assessment['level2']['pass_count']}/{assessment['level2']['max_count']} ({assessment['level2']['score_pct']:.0f}%)\n"
    md += f"- Level 3 (Stability): {assessment['level3']['pass_count']}/{assessment['level3']['max_count']} ({assessment['level3']['score_pct']:.0f}%)\n"
    md += f"- Level 4 (Learning): {assessment['level4']['pass_count']}/{assessment['level4']['max_count']} ({assessment['level4']['score_pct']:.0f}%)\n\n"
    md += f"**Total: {assessment['total_criteria_passed']}/{assessment['max_criteria']} criteria passed ({assessment['overall_score_pct']:.1f}%)**\n\n"
    
    md += "---\n\n"
    
    # Recommendations
    md += "### 💡 Recommendations\n\n"
    recs = assessment['recommendations']
    
    if recs['for_live_trading']:
        md += "**For Live Trading:**\n"
        for i, rec in enumerate(recs['for_live_trading'], 1):
            md += f"{i}. {rec}\n"
        md += "\n"
    
    if recs['risk_management']:
        md += "**Risk Management:**\n"
        for rec in recs['risk_management']:
            md += f"- {rec}\n"
        md += "\n"
    
    if recs['monitoring_plan']:
        md += "**Monitoring Plan:**\n"
        for rec in recs['monitoring_plan']:
            md += f"- {rec}\n"
        md += "\n"
    
    md += "---\n\n"
    md += f"### 📊 Confidence Level: **{assessment['confidence']}%**\n\n"
    
    # Объяснение уверенности
    if assessment['confidence'] >= 80:
        md += "**High confidence** - Model meets most quality criteria and is ready for real money trading.\n"
    elif assessment['confidence'] >= 60:
        md += "**Moderate confidence** - Model is acceptable but has some concerns. Use reduced risk.\n"
    elif assessment['confidence'] >= 40:
        md += "**Low confidence** - Model needs more validation. Paper trading recommended.\n"
    else:
        md += "**Very low confidence** - Model not ready for trading. Consider retraining.\n"
    
    return md
