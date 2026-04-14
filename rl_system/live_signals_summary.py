"""
RL ENSEMBLE LIVE SUMMARY
=========================

Агрегирует последние сигналы всех запущенных RL‑агентов и показывает
консолидированную сводку по символам (BUY / SELL / HOLD + список моделей).

Также может отправлять эту сводку в Telegram.

АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ ЗАСТРЯВШИХ МОДЕЛЕЙ
==============================================

С версии 2026-01-01 добавлено автоматическое обнаружение моделей, которые
застряли на одном типе сигнала (BUY/SELL/HOLD) более 7 дней.

Модель помечается как "STUCK", если за последние 7 дней она выдает ТОЛЬКО
один тип сигнала (100% BUY, 100% SELL или 100% HOLD) при наличии минимум
100 сигналов за этот период.

Визуальная индикация:
- Консоль: "⚠️ STUCK:BUY" / "⚠️ STUCK:SELL" / "⚠️ STUCK:HOLD"
- Telegram: "⚠️ STUCK"

Для детального анализа используйте:
    python rl_system/analyze_stuck_models.py

Подробная документация: rl_system/STUCK_MODELS_DETECTION.md

Пример запуска (через VS Code task или вручную):

    python rl_system/live_signals_summary.py --interval 60 --telegram
    python rl_system/live_signals_summary.py --symbol ADAUSDT --symbol BTCUSDT --interval 60 --telegram
    python rl_system/live_signals_summary.py --once
"""

import argparse
import json
import time
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Добавляем корень проекта в sys.path, чтобы импортировать telegram_sender
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from telegram_sender import send_telegram_message
except Exception:
    # Фоллбек: если по какой-то причине импорт не удался, просто отключаем Telegram.
    send_telegram_message = None


STATE_DIR = Path(__file__).parent / "live_state"
STATE_DIR.mkdir(exist_ok=True)

# Логи сигналов, которые пишет каждый live‑агент
SIGNALS_DIR = ROOT / "signals"
SIGNALS_DIR.mkdir(exist_ok=True)


# Кэш статистики сигналов по файлу.
# Важно: логи signals/* постоянно дописываются, поэтому кэш валидируем по mtime.
_SIGNAL_STATS_CACHE: dict[str, dict] = {}

# Кэш метрик модели (selected_best_by_metrics.json) по модели.
# Валидируем по mtime файла с метриками.
_MODEL_METRICS_CACHE: dict[str, dict] = {}

# Кэш предыдущих сигналов для отслеживания изменений
_PREVIOUS_SIGNALS: dict[str, str] = {}

# Кэш time_since_open_sec для обнаружения закрытия свечей (решение агента)
_PREVIOUS_SINCE_OPEN: dict[str, float] = {}


def load_latest_states(symbol_filters=None, max_age_seconds: int | None = None):
    """Загружает последние сигналы всех моделей из STATE_DIR.

    Возвращает словарь вида:
        {
          'ADAUSDT': {
              'BUY':  [
                  {
                      'name': 'model_name_1',
                      'time_since_open_sec': 123,
                      'time_to_close_sec': 456,
                  },
                  ...
              ],
              'SELL': [...],
              'HOLD': [...],
          },
          'BTCUSDT': {...},
          ...
        }
    """
    by_symbol = defaultdict(lambda: {"BUY": [], "SELL": [], "HOLD": []})

    now_ts = time.time()

    for file in STATE_DIR.glob("*.json"):
        # Фильтруем устаревшие сигналы по времени изменения файла
        if max_age_seconds is not None:
            try:
                mtime = file.stat().st_mtime
                if now_ts - mtime > max_age_seconds:
                    continue
            except OSError:
                continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        symbol = data.get("symbol")
        if not symbol:
            continue

        if symbol_filters and symbol not in symbol_filters:
            continue

        signal = data.get("signal", "HOLD").upper()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"

        model_info = data.get("model_info", {})
        model_name = (
            model_info.get("model_identifier")
            or model_info.get("model_filename")
            or file.stem
        )

        # В новых версиях live-агента в состоянии есть поля с секундами с
        # момента открытия свечи и до её закрытия. Для старых файлов их может
        # не быть, поэтому аккуратно берём через get.
        model_entry = {
            "name": model_name,
            "time_since_open_sec": data.get("time_since_open_sec"),
            "time_to_close_sec": data.get("time_to_close_sec"),
            "symbol": symbol,
            "timeframe": data.get("timeframe"),
            # Дополнительные поля, которые live-агент пишет в state (могут отсутствовать в старых файлах)
            "action_probabilities": data.get("action_probabilities"),
            "stop_loss": data.get("stop_loss"),
            "take_profit": data.get("take_profit"),
            "sl_multiplier": data.get("sl_multiplier"),
            "tp_multiplier": data.get("tp_multiplier"),
            "price": data.get("price"),
        }

        by_symbol[symbol][signal].append(model_entry)

    return by_symbol


def _fmt_hhmm_from_seconds(value):
    """Преобразует количество секунд в строку HH:MM.

    Если значение None или некорректно, возвращает None (без отображения).
    """
    if value is None:
        return None
    try:
        total_seconds = int(value)
    except (TypeError, ValueError):
        return None

    if total_seconds < 0:
        total_seconds = 0
    minutes, _ = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 99:
        hours = 99
    return f"{hours:02d}:{minutes:02d}"


def _fmt_float(value, digits: int = 2):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _fmt_pct(value, digits: int = 1):
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return None


def _fmt_price(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Для низких цен (например ADA ~0.x) 2 знаков недостаточно — показываем 4.
    # Для высоких цен (например BTC ~80k) оставляем 2, чтобы не раздувать вывод.
    decimals = 4 if abs(v) < 1000 else 2
    return f"{v:,.{decimals}f}"


def _fmt_action_probabilities(probs):
    """Форматирует вероятности действий модели (из live_state: action_probabilities).

    Ожидаемый формат probs:
        {'HOLD': float, 'BUY': float, 'SELL': float}
    Возвращает строку вида: p(H/B/S)=35/30/34%
    """
    if not isinstance(probs, dict):
        return None

    def _get_pct(key):
        v = probs.get(key)
        try:
            return int(round(float(v) * 100))
        except (TypeError, ValueError):
            return None

    ph = _get_pct("HOLD")
    pb = _get_pct("BUY")
    ps = _get_pct("SELL")

    if ph is None and pb is None and ps is None:
        return None

    # Если какой-то из ключей отсутствует, показываем только имеющиеся
    parts = []
    if ph is not None:
        parts.append(f"H={ph}%")
    if pb is not None:
        parts.append(f"B={pb}%")
    if ps is not None:
        parts.append(f"S={ps}%")
    return "p(" + "/".join([p.split("=")[0] for p in parts]) + ")=" + "/".join([p.split("=")[1] for p in parts])


def _get_model_selected_metrics(model_name: str | None):
    """Читает метрики обученной модели из rl_system/models/<model>/selected_best_by_metrics.json.
    
    Поддерживает два формата файла:
    1. Старый (прямые поля в корне): {"total_return_pct": 17.21, ...}
    2. Новый (вложенные в best_checkpoint): {"best_checkpoint": {"total_return_pct": 16.42, ...}}
    """
    if not model_name:
        return None

    metrics_path = ROOT / "rl_system" / "models" / model_name / "selected_best_by_metrics.json"
    if not metrics_path.exists():
        return None

    cache_key = str(metrics_path)
    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return None

    cached = _MODEL_METRICS_CACHE.get(cache_key)
    if cached is not None and cached.get("mtime") == mtime:
        return cached.get("metrics")

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Fallback: если метрики вложены в "best_checkpoint", извлекаем оттуда
    metrics = data
    if "best_checkpoint" in data and isinstance(data["best_checkpoint"], dict):
        # Новый формат: метрики находятся внутри best_checkpoint
        metrics = data["best_checkpoint"]
    # Иначе используем старый формат (метрики в корне)

    _MODEL_METRICS_CACHE[cache_key] = {"mtime": mtime, "metrics": metrics}
    return metrics


def _build_stats_from_log_file(log_path: Path, now: datetime | None = None):
    """Считает статистику сигналов (BUY/SELL/HOLD) из одного log-файла.

    Возвращает:
        {
          '24h': {'BUY': int, 'SELL': int, 'HOLD': int},
          '7d':  {'BUY': int, 'SELL': int, 'HOLD': int},
          '30d': {'BUY': int, 'SELL': int, 'HOLD': int},
        }
    """
    if now is None:
        now = datetime.now()

    stats = {
        "24h": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "7d": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "30d": {"BUY": 0, "SELL": 0, "HOLD": 0},
    }

    if not log_path.exists():
        return stats

    cutoff_24h = now - timedelta(days=1)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                sig = rec.get("signal")
                if not isinstance(sig, str):
                    continue
                sig = sig.upper()
                if sig not in ("BUY", "SELL", "HOLD"):
                    continue

                ts_str = rec.get("time") or rec.get("timestamp")
                if not isinstance(ts_str, str):
                    continue
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

                if ts >= cutoff_24h:
                    stats["24h"][sig] += 1
                if ts >= cutoff_7d:
                    stats["7d"][sig] += 1
                if ts >= cutoff_30d:
                    stats["30d"][sig] += 1
    except Exception:
        # Статистика – вспомогательная, не должна ломать сводку
        return stats

    return stats


def _get_model_stats(
    symbol: str | None,
    timeframe: str | None,
    model_name: str | None,
    now: datetime | None = None,
):
    """Возвращает статистику сигналов по модели за 24h/7d/30d.

    Используем только per-model лог-файлы (без общего лога на timeframe),
    чтобы статистика никогда не смешивалась между моделями.
    Кэш валидируем по mtime (лог постоянно дописывается).
    """
    if not symbol or not timeframe or not model_name:
        return None

    # Имя per-model log-файла должно совпадать с тем, что пишет live-агент.
    # В run_live_agent.py используется санитизация: re.sub(r"[^A-Za-z0-9._-]+", "_", model_identifier)
    safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name))

    # Только отдельный файл на модель
    per_model_log = SIGNALS_DIR / f"rl_signals_{symbol}_{timeframe}_{safe_model_id}.log"

    if not per_model_log.exists():
        return None

    log_path = per_model_log

    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        return None

    cache_key = str(log_path)
    cached = _SIGNAL_STATS_CACHE.get(cache_key)
    if cached is None or cached.get("mtime") != mtime:
        stats = _build_stats_from_log_file(log_path, now=now)
        _SIGNAL_STATS_CACHE[cache_key] = {"mtime": mtime, "stats": stats}
    else:
        stats = cached.get("stats")

    return stats


def _extract_model_creation_date(model_name: str | None):
    """Извлекает дату создания модели из имени.
    
    Формат имени: SYMBOL_TIMEFRAME_ALGO_PARAMS_YYYYMMDD_HHMMSS
    Например: BTCUSDT_1h_A2C_2000d_bt59d_20251223_154633
    
    Возвращает datetime объект или None если не удалось распарсить.
    """
    if not model_name:
        return None
    
    # Ищем паттерн: 8 цифр (дата) + _ + 6 цифр (время)
    import re
    match = re.search(r'(\d{8})_(\d{6})', str(model_name))
    if not match:
        return None
    
    date_str = match.group(1)  # YYYYMMDD
    time_str = match.group(2)  # HHMMSS
    
    try:
        from datetime import datetime
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _check_if_model_stuck(stats: dict | None, model_name: str | None = None, threshold_days: int = 7):
    """Проверяет, застряла ли модель на одном типе сигнала за последние N дней.
    
    ВАЖНО: Модель должна работать минимум threshold_days дней, прежде чем мы
    сможем определить её как "застрявшую". Это предотвращает ложные срабатывания
    для только что обученных моделей.
    
    Args:
        stats: Статистика сигналов модели (24h/7d/30d)
        model_name: Имя модели для извлечения даты создания
        threshold_days: Порог в днях (по умолчанию 7)
    
    Возвращает:
        - None: если статистика недоступна или недостаточно данных
        - "BUY": если модель дает только BUY сигналы
        - "SELL": если модель дает только SELL сигналы
        - "HOLD": если модель дает только HOLD сигналы
        - False: если модель нормально меняет сигналы
    """
    if not isinstance(stats, dict):
        return None
    
    # Проверяем возраст модели - должна работать минимум threshold_days дней
    creation_date = _extract_model_creation_date(model_name)
    if creation_date is not None:
        from datetime import datetime, timedelta
        now = datetime.now()
        model_age_days = (now - creation_date).total_seconds() / 86400  # в днях
        
        # Если модель работает меньше порогового периода, не проверяем на "застревание"
        if model_age_days < threshold_days:
            return None
    
    window = "7d" if threshold_days == 7 else "30d"
    d = stats.get(window, {}) if isinstance(stats.get(window), dict) else {}
    
    b = int(d.get("BUY", 0) or 0)
    s = int(d.get("SELL", 0) or 0)
    h = int(d.get("HOLD", 0) or 0)
    
    total = b + s + h
    
    # Если данных слишком мало (< 100 сигналов за 7 дней), не считаем это проблемой
    if total < 100:
        return None
    
    # НОВАЯ ЛОГИКА: Модель застряла, если она НЕ меняет между BUY и SELL
    # Хорошая модель: периодически дает и BUY, и SELL (независимо от HOLD)
    # Застрявшая модель: только BUY+HOLD (без SELL) или только SELL+HOLD (без BUY) или только HOLD
    
    if b > 0 and s > 0:
        # Модель меняет между BUY и SELL → работает нормально
        return False
    
    # Модель не дает оба направленных сигнала → застряла
    if b > 0 and s == 0:
        # Только BUY (может быть с HOLD)
        return "BUY"
    elif s > 0 and b == 0:
        # Только SELL (может быть с HOLD)
        return "SELL"
    elif h > 0 and b == 0 and s == 0:
        # Только HOLD
        return "HOLD"
    
    return False


def _format_signal_stats_compact(stats: dict | None):
    """Форматирует статистику сигналов в компактный вид: 24h:B=.../S=.../H=...; 7d:...; 30d:..."""
    if not isinstance(stats, dict):
        return None

    def _fmt_window(label: str):
        d = stats.get(label, {}) if isinstance(stats.get(label), dict) else {}
        b = int(d.get("BUY", 0) or 0)
        s = int(d.get("SELL", 0) or 0)
        h = int(d.get("HOLD", 0) or 0)
        return f"{label}:B={b}/S={s}/H={h}"

    return "; ".join([_fmt_window("24h"), _fmt_window("7d"), _fmt_window("30d")])


def _detect_signal_changes(by_symbol):
    """Обнаруживает изменения сигналов моделей.
    
    Возвращает список словарей с информацией об изменениях:
    [
        {
            'symbol': 'ADAUSDT',
            'model_name': 'ADAUSDT_15m_A2C_800d...',
            'old_signal': 'HOLD',
            'new_signal': 'BUY',
            'timestamp': '2026-01-02 17:38:45'
        },
        ...
    ]
    """
    global _PREVIOUS_SIGNALS
    
    changes = []
    current_signals = {}
    
    # Собираем текущие сигналы всех моделей
    for symbol, stats in by_symbol.items():
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            for model in stats[signal_type]:
                model_name = model.get('name', '<unknown>')
                if model_name == '<unknown>':
                    continue
                
                # Уникальный ключ: symbol + model_name
                key = f"{symbol}_{model_name}"
                current_signals[key] = signal_type
                
                # Проверяем, изменился ли сигнал
                old_signal = _PREVIOUS_SIGNALS.get(key)
                if old_signal is not None and old_signal != signal_type:
                    changes.append({
                        'symbol': symbol,
                        'model_name': model_name,
                        'old_signal': old_signal,
                        'new_signal': signal_type,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
    
    # Обновляем кэш предыдущих сигналов
    _PREVIOUS_SIGNALS = current_signals
    
    return changes


def _detect_candle_close_events(by_symbol, update_interval_seconds: int = 60):
    """Обнаруживает момент принятия решения агентом (закрытие свечи / открытие новой свечи).

    Принцип: time_since_open_sec монотонно растёт внутри одной свечи.
    Если в текущей итерации значение оказалось значительно меньше предыдущего
    (сброс счётчика), значит открылась новая свеча — агент принял решение.

    Returns:
        Список словарей с описанием событий закрытия свечи.
    """
    global _PREVIOUS_SINCE_OPEN

    events = []

    for symbol, signal_groups in by_symbol.items():
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            for model in signal_groups[signal_type]:
                model_name = model.get('name', '<unknown>')
                if model_name == '<unknown>':
                    continue

                since_open = model.get('time_since_open_sec')
                if since_open is None:
                    continue

                try:
                    since_open = float(since_open)
                except (TypeError, ValueError):
                    continue

                key = f"{symbol}_{model_name}"
                prev = _PREVIOUS_SINCE_OPEN.get(key)

                if prev is not None:
                    # Если счётчик сбросился (новое значение намного меньше предыдущего),
                    # значит открылась новая свеча → агент принял решение на закрытии предыдущей.
                    # Порог: новое значение < предыдущее - (interval * 0.5)
                    if since_open < prev - update_interval_seconds * 0.5:
                        events.append({
                            'symbol': symbol,
                            'model_name': model_name,
                            'signal': signal_type,
                            'timeframe': model.get('timeframe'),
                            'model': model,
                        })

                _PREVIOUS_SINCE_OPEN[key] = since_open

    return events


def _play_candle_close_sound(events: list):
    """Воспроизводит звуковой сигнал при закрытии свечи.

    Тональность зависит от преобладающего сигнала:
    - BUY  → восходящие тоны (оптимистичный)
    - SELL → нисходящие тоны (предупреждение)
    - HOLD → нейтральный одиночный тон
    """
    if not events:
        return

    try:
        import winsound
    except ImportError:
        return

    counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    for e in events:
        sig = e.get('signal', 'HOLD')
        counts[sig] = counts.get(sig, 0) + 1

    dominant = max(counts, key=lambda k: counts[k])

    try:
        if dominant == 'BUY':
            winsound.Beep(600, 80)
            winsound.Beep(900, 80)
            winsound.Beep(1200, 180)
        elif dominant == 'SELL':
            winsound.Beep(1200, 80)
            winsound.Beep(900, 80)
            winsound.Beep(600, 180)
        else:  # HOLD
            winsound.Beep(750, 200)
    except Exception:
        pass


def _format_candle_close_telegram(events: list):
    """Форматирует Telegram-уведомление о принятии решения агентами на закрытии свечи."""
    if not events:
        return None

    now_str = datetime.now().strftime("%H:%M:%S")
    lines = [f"\U0001f56f\ufe0f <b>CANDLE CLOSE DECISION</b>  {now_str}", ""]

    # Группируем по символу
    by_sym: dict = {}
    for e in events:
        sym = e['symbol']
        by_sym.setdefault(sym, []).append(e)

    for symbol in sorted(by_sym.keys()):
        for e in by_sym[symbol]:
            signal = e['signal']
            tf = e.get('timeframe') or '?'
            model_name = e.get('model_name', '<unknown>')
            model = e.get('model', {})

            sig_emoji = '\U0001f7e2' if signal == 'BUY' else '\U0001f534' if signal == 'SELL' else '\u26aa'
            lines.append(f"{sig_emoji} <b>{symbol}</b> ({tf})  \u2192  <b>{signal}</b>")

            probs = model.get('action_probabilities')
            if probs:
                prob_str = _fmt_action_probabilities(probs)
                if prob_str:
                    lines.append(f"   {prob_str}")

            metrics = _get_model_selected_metrics(model_name)
            mm_str = _format_model_metrics_compact(metrics)
            if mm_str:
                lines.append(f"   {mm_str}")

            short = model_name[:55] + "\u2026" if len(model_name) > 55 else model_name
            lines.append(f"   <code>{short}</code>")
            lines.append("")

    return "\n".join(lines).rstrip()


def _format_model_metrics_compact(metrics: dict | None):
    """Компактно форматирует метрики модели из selected_best_by_metrics.json."""
    if not isinstance(metrics, dict):
        return None

    parts: list[str] = []

    # Основные метрики (названия полей — как в selected_best_by_metrics.json)
    wr = _fmt_pct(metrics.get("win_rate_pct"), digits=1)
    pf = _fmt_float(metrics.get("profit_factor"), digits=2)
    tr = _fmt_pct(metrics.get("total_return_pct"), digits=2)
    trades = metrics.get("total_trades")
    dd = _fmt_pct(metrics.get("max_drawdown_pct"), digits=2)
    score = _fmt_float(metrics.get("score"), digits=3)

    if wr is not None:
        parts.append(f"WR={wr}")
    if pf is not None:
        parts.append(f"PF={pf}")
    if tr is not None:
        parts.append(f"Ret={tr}")
    if trades is not None:
        try:
            parts.append(f"Trades={int(trades)}")
        except (TypeError, ValueError):
            pass
    if dd is not None:
        parts.append(f"DD={dd}")
    if score is not None:
        parts.append(f"Score={score}")

    return ", ".join(parts) if parts else None


def format_console_summary(by_symbol):
    """Формирует строковое представление сводки для консоли."""
    lines = []
    now_dt = datetime.now()
    timestamp = now_dt.strftime("%H:%M:%S")
    lines.append("=" * 70)
    lines.append(f"{timestamp} | RL ENSEMBLE SUMMARY")

    if not by_symbol:
        lines.append("No live signals found in live_state/ directory.")
        return "\n".join(lines)

    for symbol in sorted(by_symbol.keys()):
        stats = by_symbol[symbol]
        buy_models = stats["BUY"]
        sell_models = stats["SELL"]
        hold_models = stats["HOLD"]

        # Строка с агрегированными количествами + иконки
        line = (
            f"{timestamp} | {symbol} | "
            f"🟢 BUY: {len(buy_models)} | "
            f"🔴 SELL: {len(sell_models)} | "
            f"⚪ HOLD: {len(hold_models)}"
        )
        lines.append(line)

        # Отдельные блоки по каждому сигналу, модели построчно
        def _sorted_models(models):
            """Сортируем модели по свежести сигнала (since_open), затем по имени.

            Более свежие (меньше time_since_open_sec) идут выше. Если тайминга
            нет, такие модели отправляем в конец списка и сортируем по имени.
            """

            def sort_key(m):
                so_raw = m.get("time_since_open_sec")
                try:
                    so_val = int(so_raw) if so_raw is not None else None
                except (TypeError, ValueError):
                    so_val = None

                # has_time = 0 для моделей с валидным since_open, 1 – без него
                has_time = 0 if so_val is not None else 1
                # Для моделей без времени кладём большое число, чтобы они были внизу
                so_for_sort = so_val if so_val is not None else 10**9
                name = m.get("name", "")
                return (has_time, so_for_sort, name)

            return sorted(models, key=sort_key)

        if buy_models:
            lines.append("   🟢 BUY:")
            for m in _sorted_models(buy_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)

                # Проверка на "застрявшую" модель
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK:{stuck_signal}"

                details_parts: list[str] = []
                if so is not None:
                    details_parts.append(f"since_open={so}")
                if tc is not None:
                    details_parts.append(f"to_close={tc}")

                ap = _fmt_action_probabilities(m.get("action_probabilities"))
                if ap:
                    details_parts.append("\n" + ap)

                sl = _fmt_price(m.get("stop_loss"))
                tp = _fmt_price(m.get("take_profit"))
                slm = _fmt_float(m.get("sl_multiplier"), digits=1)
                tpm = _fmt_float(m.get("tp_multiplier"), digits=1)
                if sl is not None or tp is not None:
                    tp_sl_parts = []
                    if tp is not None:
                        tp_sl_parts.append(f"\nTP={tp}")
                    if sl is not None:
                        tp_sl_parts.append(f"SL={sl}")
                    if tpm is not None or slm is not None:
                        mult = []
                        if tpm is not None:
                            mult.append(f"tp_mult={tpm}")
                        if slm is not None:
                            mult.append(f"sl_mult={slm}")
                        if mult:
                            tp_sl_parts.append("(" + ", ".join(mult) + ")")
                    details_parts.append(" ".join(tp_sl_parts))

                stats_str = _format_signal_stats_compact(stats)
                if stats_str:
                    details_parts.append("\n" + stats_str)

                model_metrics = _get_model_selected_metrics(name)
                mm_str = _format_model_metrics_compact(model_metrics)
                if mm_str:
                    details_parts.append("\n" + mm_str + "\n")

                lines.append(f"      - {name}{stuck_indicator}")
                if details_parts:
                    lines.append(", ".join(details_parts))

        if sell_models:
            lines.append("   🔴 SELL:")
            for m in _sorted_models(sell_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)

                # Проверка на "застрявшую" модель
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK:{stuck_signal}"

                details_parts: list[str] = []
                if so is not None:
                    details_parts.append(f"since_open={so}")
                if tc is not None:
                    details_parts.append(f"to_close={tc}")

                ap = _fmt_action_probabilities(m.get("action_probabilities"))
                if ap:
                    details_parts.append("\n" + ap)

                sl = _fmt_price(m.get("stop_loss"))
                tp = _fmt_price(m.get("take_profit"))
                slm = _fmt_float(m.get("sl_multiplier"), digits=1)
                tpm = _fmt_float(m.get("tp_multiplier"), digits=1)
                if sl is not None or tp is not None:
                    tp_sl_parts = []
                    if tp is not None:
                        tp_sl_parts.append(f"\nTP={tp}")
                    if sl is not None:
                        tp_sl_parts.append(f"SL={sl}")
                    if tpm is not None or slm is not None:
                        mult = []
                        if tpm is not None:
                            mult.append(f"tp_mult={tpm}")
                        if slm is not None:
                            mult.append(f"sl_mult={slm}")
                        if mult:
                            tp_sl_parts.append("(" + ", ".join(mult) + ")")
                    details_parts.append(" ".join(tp_sl_parts))

                stats_str = _format_signal_stats_compact(stats)
                if stats_str:
                    details_parts.append("\n" + stats_str)

                model_metrics = _get_model_selected_metrics(name)
                mm_str = _format_model_metrics_compact(model_metrics)
                if mm_str:
                    details_parts.append("\n" + mm_str + "\n")

                lines.append(f"      - {name}{stuck_indicator}")
                if details_parts:
                    lines.append(", ".join(details_parts))

        if hold_models:
            lines.append("   ⚪ HOLD:")
            for m in _sorted_models(hold_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)

                # Проверка на "застрявшую" модель
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK:{stuck_signal}"

                details_parts: list[str] = []
                if so is not None:
                    details_parts.append(f"since_open={so}")
                if tc is not None:
                    details_parts.append(f"to_close={tc}")

                ap = _fmt_action_probabilities(m.get("action_probabilities"))
                if ap:
                    details_parts.append("\n" + ap)

                sl = _fmt_price(m.get("stop_loss"))
                tp = _fmt_price(m.get("take_profit"))
                slm = _fmt_float(m.get("sl_multiplier"), digits=1)
                tpm = _fmt_float(m.get("tp_multiplier"), digits=1)
                if sl is not None or tp is not None:
                    tp_sl_parts = []
                    if tp is not None:
                        tp_sl_parts.append(f"\nTP={tp}")
                    if sl is not None:
                        tp_sl_parts.append(f"SL={sl}")
                    if tpm is not None or slm is not None:
                        mult = []
                        if tpm is not None:
                            mult.append(f"tp_mult={tpm}")
                        if slm is not None:
                            mult.append(f"sl_mult={slm}")
                        if mult:
                            tp_sl_parts.append("(" + ", ".join(mult) + ")")
                    details_parts.append(" ".join(tp_sl_parts))

                stats_str = _format_signal_stats_compact(stats)
                if stats_str:
                    details_parts.append("\n" + stats_str)

                model_metrics = _get_model_selected_metrics(name)
                mm_str = _format_model_metrics_compact(model_metrics)
                if mm_str:
                    details_parts.append("\n" + mm_str + "\n")

                lines.append(f"      - {name}{stuck_indicator}")
                if details_parts:
                    lines.append(", ".join(details_parts))

    # Добавляем информацию об изменениях сигналов ПЕРЕД итоговой сводкой
    signal_changes = _detect_signal_changes(by_symbol)
    
    if signal_changes:
        lines.append("-" * 70)
        lines.append("🔔 SIGNAL CHANGES:")
        for change in signal_changes:
            old_emoji = "🟢" if change['old_signal'] == 'BUY' else "🔴" if change['old_signal'] == 'SELL' else "⚪"
            new_emoji = "🟢" if change['new_signal'] == 'BUY' else "🔴" if change['new_signal'] == 'SELL' else "⚪"
            
            lines.append(
                f"   {change['symbol']} | {change['model_name'][:50]}{'...' if len(change['model_name']) > 50 else ''}"
            )
            lines.append(
                f"      {old_emoji} {change['old_signal']} → {new_emoji} {change['new_signal']} at {change['timestamp']}"
            )

    # В конце дублируем компактную сводку по всем символам,
    # чтобы внизу консоли всегда было видно итоговые количества.
    lines.append("-" * 70)
    for symbol in sorted(by_symbol.keys()):
        stats = by_symbol[symbol]
        lines.append(
            f"{symbol} | 🟢 BUY: {len(stats['BUY'])} | 🔴 SELL: {len(stats['SELL'])} | ⚪ HOLD: {len(stats['HOLD'])}"
        )

    return "\n".join(lines)


def format_telegram_summary(by_symbol):
    """Формирует текст для отправки в Telegram."""
    if not by_symbol:
        return "📊 RL Ensemble Summary\n\nNo live signals found."

    now_dt = datetime.now()

    # В Telegram отправляем только компактную сводку (без расширенных деталей),
    # иначе сообщение легко превышает лимит Telegram (~4096 символов).
    lines = ["📊 <b>RL ENSEMBLE SUMMARY</b>"]

    for symbol in sorted(by_symbol.keys()):
        stats = by_symbol[symbol]
        buy_models = stats["BUY"]
        sell_models = stats["SELL"]
        hold_models = stats["HOLD"]

        lines.append("")
        lines.append(f"<b>{symbol}</b>")
        lines.append(
            f"🟢 BUY: <b>{len(buy_models)}</b> | "
            f"🔴 SELL: <b>{len(sell_models)}</b> | "
            f"⚪ HOLD: <b>{len(hold_models)}</b>"
        )

        def _sorted_models(models):
            """Та же сортировка, что и для консоли: по свежести, затем по имени."""

            def sort_key(m):
                so_raw = m.get("time_since_open_sec")
                try:
                    so_val = int(so_raw) if so_raw is not None else None
                except (TypeError, ValueError):
                    so_val = None

                has_time = 0 if so_val is not None else 1
                so_for_sort = so_val if so_val is not None else 10**9
                name = m.get("name", "")
                return (has_time, so_for_sort, name)

            return sorted(models, key=sort_key)

        if buy_models:
            lines.append("🟢 BUY:")
            for m in _sorted_models(buy_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                
                # Получаем статистику для проверки "застрявшей" модели
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK"
                
                if so is not None and tc is not None:
                    lines.append(f"• {name}{stuck_indicator} (since_open={so}, to_close={tc})")
                else:
                    lines.append(f"• {name}{stuck_indicator}")

        if sell_models:
            lines.append("🔴 SELL:")
            for m in _sorted_models(sell_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                
                # Получаем статистику для проверки "застрявшей" модели
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK"
                
                if so is not None and tc is not None:
                    lines.append(f"• {name}{stuck_indicator} (since_open={so}, to_close={tc})")
                else:
                    lines.append(f"• {name}{stuck_indicator}")

        if hold_models:
            lines.append("⚪ HOLD:")
            for m in _sorted_models(hold_models):
                name = m.get("name", "<unknown>")
                so = _fmt_hhmm_from_seconds(m.get("time_since_open_sec"))
                tc = _fmt_hhmm_from_seconds(m.get("time_to_close_sec"))
                
                # Получаем статистику для проверки "застрявшей" модели
                stats = _get_model_stats(m.get("symbol"), m.get("timeframe"), name, now=now_dt)
                stuck_signal = _check_if_model_stuck(stats, model_name=name, threshold_days=7)
                stuck_indicator = ""
                if stuck_signal and stuck_signal != False:
                    stuck_indicator = f" ⚠️ STUCK"
                
                if so is not None and tc is not None:
                    lines.append(f"• {name}{stuck_indicator} (since_open={so}, to_close={tc})")
                else:
                    lines.append(f"• {name}{stuck_indicator}")

    # Добавляем информацию об изменениях сигналов
    signal_changes = _detect_signal_changes(by_symbol)
    
    if signal_changes:
        lines.append("")
        lines.append("🔔 <b>SIGNAL CHANGES:</b>")
        for change in signal_changes[:10]:  # Ограничиваем до 10 изменений для Telegram
            old_emoji = "🟢" if change['old_signal'] == 'BUY' else "🔴" if change['old_signal'] == 'SELL' else "⚪"
            new_emoji = "🟢" if change['new_signal'] == 'BUY' else "🔴" if change['new_signal'] == 'SELL' else "⚪"
            
            model_short = change['model_name'][:40] + "..." if len(change['model_name']) > 40 else change['model_name']
            lines.append(
                f"• {change['symbol']}: {model_short}"
            )
            lines.append(
                f"  {old_emoji} {change['old_signal']} → {new_emoji} {change['new_signal']}"
            )
        
        if len(signal_changes) > 10:
            lines.append(f"  ... and {len(signal_changes) - 10} more changes")

    # Защита от лимита Telegram: обрезаем по строкам, сохраняя валидность HTML.
    max_len = 3900
    while len("\n".join(lines)) > max_len and len(lines) > 3:
        lines.pop()
    if len("\n".join(lines)) > max_len:
        # Совсем крайний случай: отправим только заголовок
        return "📊 <b>RL ENSEMBLE SUMMARY</b>"

    # Если пришлось обрезать — добавим пометку
    text = "\n".join(lines)
    if len(text) > max_len - 20:
        text = text[: max_len - 20].rstrip()
    return text


def main():
    parser = argparse.ArgumentParser(
        description="RL Ensemble live signals summary",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="Filter by symbol (e.g. ADAUSDT). Can be specified multiple times.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (for continuous mode)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send ensemble summary to Telegram on each update.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only once and exit (no loop).",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=3600,
        help=(
            "Ignore live_state files older than this many seconds. "
            "Set to 0 to disable age filtering. Default: 3600 (1 hour)."
        ),
    )
    parser.add_argument(
        "--candle-close-sound",
        action="store_true",
        help="Play sound when any model makes a new decision (candle close).",
    )
    parser.add_argument(
        "--candle-close-telegram",
        action="store_true",
        help="Send Telegram notification when any model makes a new decision (candle close).",
    )

    args = parser.parse_args()

    symbols = args.symbol if args.symbol else None

    max_age = None if args.max_age_seconds == 0 else args.max_age_seconds

    def run_once():
        by_symbol = load_latest_states(symbol_filters=symbols, max_age_seconds=max_age)

        # Обнаруживаем события закрытия свечей (до форматирования сводки)
        candle_events = _detect_candle_close_events(by_symbol, update_interval_seconds=args.interval)

        console_text = format_console_summary(by_symbol)
        print(console_text)

        # Звук при закрытии свечи
        if args.candle_close_sound and candle_events:
            _play_candle_close_sound(candle_events)
            print(f"\U0001f514 Candle close: {len(candle_events)} model(s) made a decision")
            for e in candle_events:
                sig_emoji = '\U0001f7e2' if e['signal'] == 'BUY' else '\U0001f534' if e['signal'] == 'SELL' else '\u26aa'
                tf = e.get('timeframe') or '?'
                print(f"   {sig_emoji} {e['symbol']} ({tf}) \u2192 {e['signal']}  [{e['model_name'][:50]}]")

        # Telegram при закрытии свечи
        if args.candle_close_telegram and candle_events and send_telegram_message is not None:
            try:
                tg_text = _format_candle_close_telegram(candle_events)
                if tg_text:
                    ok = send_telegram_message(tg_text)
                    if not ok:
                        print("\u26a0\ufe0f Candle close Telegram send returned False")
            except Exception as e:
                print(f"\u274c Candle close Telegram error: {e}")

        if args.telegram and send_telegram_message is not None:
            try:
                tg_text = format_telegram_summary(by_symbol)
                ok = send_telegram_message(tg_text)
                if not ok:
                    print("⚠️ Telegram summary send returned False")
            except Exception as e:
                print(f"❌ Telegram summary error: {e}")

    if args.once:
        run_once()
        return

    # Непрерывный режим
    while True:
        run_once()
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Ensemble summary stopped by user")
            break


if __name__ == "__main__":
    main()
