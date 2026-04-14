"""
Модуль для отправки торговых сигналов в Telegram
"""
import requests
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Новые расчёты метрик
try:
    from risk_metrics import compute_signal_metrics, format_metrics_block, load_risk_config
except Exception:
    compute_signal_metrics = None  # graceful fallback
    format_metrics_block = None
    load_risk_config = None

# Загружаем переменные окружения
load_dotenv()

# Telegram настройки из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Проверяем, что секреты загружены
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Не найдены Telegram credentials в .env файле!")

def send_telegram_message(message: str) -> bool:
    """Отправка сообщения в Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            logging.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {str(e)}")
        return False

def format_price(price: float, symbol: str = "") -> str:
    """Умное форматирование цены в зависимости от величины"""
    if price == 0:
        return "$0.00"
    
    # Для очень маленьких цен (< 0.01) показываем больше знаков
    if price < 0.01:
        # Находим первую значащую цифру после нуля
        if price < 0.0001:
            return f"${price:.8f}"  # 8 знаков для очень мелких монет
        elif price < 0.001:
            return f"${price:.6f}"  # 6 знаков
        else:
            return f"${price:.4f}"  # 4 знака
    elif price < 1:
        return f"${price:.4f}"  # 4 знака для центов
    elif price < 100:
        return f"${price:,.3f}"  # 3 знака с разделителями
    else:
        return f"${price:,.2f}"  # 2 знака с разделителями для больших цен

def get_timeout_info(timeframe: str, horizon_bars: int) -> str:
    """Рассчитывает информацию о таймауте выхода из сделки"""
    # Преобразуем таймфрейм в минуты
    tf_minutes = {
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    minutes_per_bar = tf_minutes.get(timeframe, 30)  # По умолчанию 30 минут
    total_minutes = minutes_per_bar * horizon_bars
    
    # Форматируем время
    if total_minutes < 60:
        time_str = f"{total_minutes} мин"
    else:
        hours = total_minutes / 60
        if hours < 24:
            time_str = f"{hours:.1f} ч"
        else:
            days = hours / 24
            time_str = f"{days:.1f} дн"
    
    return f"{horizon_bars} свечей × {minutes_per_bar} мин = {time_str}"

def format_trading_signal(signal_data: dict) -> str:
    """Форматирование торгового сигнала для Telegram"""
    symbol = signal_data.get('symbol', 'Unknown')
    signal_type = signal_data.get('signal', 'HOLD')
    
    # Нормализуем сигналы: BUY -> LONG, SELL -> SHORT
    if signal_type == 'BUY':
        signal_type = 'LONG'
    elif signal_type == 'SELL':
        signal_type = 'SHORT'
    
    # Новый формат: используем 'confidence' или старый 'probs'
    confidence = signal_data.get('confidence', 0)
    if confidence == 0 and 'probs' in signal_data:
        probs = signal_data.get('probs', {})
        confidence = max(probs.values()) if probs else 0
    
    # Новый формат: используем 'price' или старый 'close'
    close_price = signal_data.get('price', signal_data.get('close', 0))
    
    timeframe = signal_data.get('timeframe', 'Unknown')
    
    # Новый формат: используем 'timestamp' или старый 'time_of_event'
    time_event = signal_data.get('timestamp', signal_data.get('time_of_event', 'Unknown'))
    
    # Форматируем время если это datetime объект
    if hasattr(time_event, 'strftime'):
        time_event = time_event.strftime('%Y-%m-%d %H:%M:%S')
    elif str(time_event).count('-') >= 2:  # Если это строка с датой
        time_event = str(time_event)[:19]  # Обрезаем до секунд
    
    # Emoji для типа сигнала
    signal_emoji = {
        'LONG': '🟢',
        'SHORT': '🔴', 
        'HOLD': '⚪'
    }.get(signal_type, '⚪')
    
    message = f"{signal_emoji} <b>{signal_type} сигнал: {symbol}</b>\n"
    message += f"💰 Цена: {format_price(close_price, symbol)}\n"
    message += f"⚡ Таймфрейм: {timeframe}\n"
    message += f"🎯 Уверенность: {confidence:.1%}\n"
    message += f"🕐 Время: {time_event}\n"
    
    # Добавляем информацию о модели: сначала предпочтительный идентификатор (папка), затем файл
    model_identifier = signal_data.get('model_identifier')
    model_filename = signal_data.get('model_filename')
    if model_identifier:
        message += f"📄 Модель: <code>{model_identifier}</code>\n"
        if model_filename:
            message += f"📄 Файл: <code>{model_filename}</code>\n"
    elif model_filename:
        message += f"📄 Модель: <code>{model_filename}</code>\n"
    
    # Добавляем детали распределения вероятностей если есть
    details = signal_data.get('details', {})
    if details:
        message += f"\n📊 <b>Детали:</b>\n"
        if 'LONG' in details:
            message += f"🟢 LONG: {details['LONG']:.1%}\n"
        if 'SHORT' in details:
            message += f"🔴 SHORT: {details['SHORT']:.1%}\n"
        if 'HOLD' in details:
            message += f"⚪ HOLD: {details['HOLD']:.1%}\n"
    
    if signal_type in ['LONG', 'SHORT']:
        sl = signal_data.get('stop_loss')
        tp = signal_data.get('take_profit')
        if tp and sl:
            message += f"\n🎯 Take Profit: {format_price(tp, symbol)}\n"
            message += f"🛡️ Stop Loss: {format_price(sl, symbol)}\n"
        
        # Добавляем статистику бэктеста
        backtest_stats = signal_data.get('backtest_stats', {})
        if backtest_stats:
            total_trades = backtest_stats.get('total_trades', 0)
            tp_count = backtest_stats.get('tp_count', 0)
            sl_count = backtest_stats.get('sl_count', 0)
            time_exit_count = backtest_stats.get('time_exit_count', 0)
            win_rate = backtest_stats.get('win_rate', 0)
            profit_factor = backtest_stats.get('profit_factor', 0)
            starting_equity = backtest_stats.get('starting_equity', 0)
            total_pnl_usd = backtest_stats.get('total_pnl_usd', 0)
            
            # Данные по выходам
            exit_breakdown = backtest_stats.get('exit_pnl_breakdown', {})
            by_exit = exit_breakdown.get('by_exit_reason', {})
            
            # Данные по Stop Loss
            sl_data = by_exit.get('stop_loss', {})
            sl_gross_loss = sl_data.get('gross_loss', 0)
            
            # Данные по Take Profit
            tp_data = by_exit.get('take_profit', {})
            tp_gross_profit = tp_data.get('gross_profit', 0)
            
            # Данные по Time Exit
            te_data = by_exit.get('time_exit', {})
            te_gross_profit = te_data.get('gross_profit', 0)
            te_gross_loss = te_data.get('gross_loss', 0)
            te_net = te_gross_profit - te_gross_loss
            
            if total_trades > 0:
                message += f"\n📈 <b>Статистика бэктеста:</b>\n"
                message += f"Всего сделок: {total_trades}\n"
                message += f"✅ TP: {tp_count} ({tp_count/total_trades*100:.1f}%)\n"
                message += f"❌ SL: {sl_count} ({sl_count/total_trades*100:.1f}%)\n"
                message += f"⏱️ Timeout: {time_exit_count} ({time_exit_count/total_trades*100:.1f}%)\n"
                message += f"\n📊 Win Rate: {win_rate:.1%}\n"
                message += f"📊 Profit Factor: {profit_factor:.2f}\n"
                message += f"\n💰 <b>P&L по типам выхода:</b>\n"
                message += f"✅ TP Profit: +${tp_gross_profit:,.2f}\n"
                message += f"❌ SL Loss: -${sl_gross_loss:,.2f}\n"
                message += f"⏱️ Timeout P&L: {'+-'[te_net<0]}${abs(te_net):,.2f}\n"
                message += f"\n💼 Начальный депозит: ${starting_equity:,.2f}\n"
                message += f"💵 Total P&L: {'+-'[total_pnl_usd<0]}${abs(total_pnl_usd):,.2f}\n"
        
        # 🛡️ Добавляем информацию о SMC фильтре
        smc_filter_info = signal_data.get('smc_filter', {})
        if smc_filter_info and smc_filter_info.get('enabled'):
            smc_result = smc_filter_info.get('result')
            if smc_result:
                confluence_score = smc_result.get('confluence_score', 0)
                approved = smc_result.get('approved', False)
                recommendation = smc_result.get('recommendation', '')
                
                message += f"\n🛡️ <b>SMC Фильтр:</b> {'✅ ОДОБРЕН' if approved else '❌ ОТКЛОНЁН'}\n"
                message += f"Confluence: {confluence_score}/6\n"
                
                # Показываем только подтвержденные confluence факторы
                reasons = smc_result.get('reasons', [])
                confirmed_reasons = [r for r in reasons if '✅' in r]
                if confirmed_reasons:
                    message += f"<b>Подтверждения:</b>\n"
                    for reason in confirmed_reasons:
                        # Убираем галочку и форматируем
                        clean_reason = reason.replace('✅ ', '')
                        message += f"  • {clean_reason}\n"
        
        # Проверяем, является ли это RL агентом
        # RL агент определяется по наличию model_filename или по вероятностям HOLD/LONG/SHORT в details
        details = signal_data.get('details', {})
        is_rl_agent = (
            'model_filename' in signal_data or
            (isinstance(details, dict) and ('HOLD' in details or 'LONG' in details or 'SHORT' in details))
        )
        
        # Добавляем информацию о таймауте только для XGBoost моделей
        if not is_rl_agent:
            horizon_bars = signal_data.get('horizon_bars', 6)  # По умолчанию 6 свечей
            if horizon_bars:
                timeout_info = get_timeout_info(timeframe, horizon_bars)
                if timeout_info:
                    message += f"\n⏱️ <b>Выход по таймауту:</b> {timeout_info}\n"

            # Добавляем блок метрик риска / EV только для XGBoost
            if compute_signal_metrics and format_metrics_block:
                try:
                    metrics = compute_signal_metrics(signal_data)
                    metrics_block = format_metrics_block(metrics)
                    if metrics_block:
                        message += metrics_block
                except Exception as e:
                    message += f"\n⚠ Ошибка расчёта метрик: {e}"
            
            # Добавляем пояснения к основным метрикам риска только для XGBoost
            message += (
                "\n\n<i>Пояснения к метрикам:</i>"
                "\n<b>p_be</b> — вероятность безубыточности (break-even), при которой сделка не убыточна."
                "\n<b>p_dir</b> — вероятность движения в нужную сторону (по мнению модели)."
                "\n<b>p_hit</b> — калиброванная вероятность достижения TP."
                "\n<b>EV(naive)</b> — ожидаемая доходность без калибровки вероятности."
                "\n<b>EV(cal)</b> — ожидаемая доходность с учётом калибровки p_hit."
                "\n<b>R:R</b> — соотношение цель/риск (Risk:Reward)."
                "\n<b>NetR:R</b> — скорректированное R:R с учётом комиссий."
            )
    return message

def send_trading_signal(signal_data: dict) -> bool:
    """Отправка торгового сигнала в Telegram"""
    signal_type = signal_data.get('signal', 'HOLD')
    
    # Отправляем только торговые сигналы (LONG/SHORT или BUY/SELL)
    if signal_type in ['LONG', 'SHORT', 'BUY', 'SELL']:
        message = format_trading_signal(signal_data)
        success = send_telegram_message(message)
        
        if success:
            # Убираем консольный лог об успешной отправке, чтобы не спамить live‑терминал.
            pass
        else:
            # Ошибку отправки оставляем видимой.
            print(f"❌ Ошибка отправки сигнала {signal_type} для {signal_data.get('symbol')} в Telegram")
            
        return success
    
    return True  # Для HOLD сигналов возвращаем True (не отправляем, но это не ошибка)


def send_heartbeat(active_models: list, uptime: str = None) -> bool:
    """Отправка heartbeat сообщения о статусе торговой системы"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"<b>Торговая система работает</b>\n"
        message += f"{current_time}\n"
        
        if uptime:
            message += f"Время работы: {uptime}\n"
        
        message += f"\n<b>Активные модели ({len(active_models)}):</b>\n"
        for model in active_models:
            message += f"▫️ {model}\n"
        
        success = send_telegram_message(message)
        
        if success:
            print(f"Heartbeat отправлен в Telegram ({len(active_models)} моделей)")
        else:
            print(f"❌ Ошибка отправки heartbeat в Telegram")
            
        return success
        
    except Exception as e:
        print(f"❌ Ошибка heartbeat: {e}")
        return False
