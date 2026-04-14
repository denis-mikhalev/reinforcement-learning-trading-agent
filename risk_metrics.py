"""Utility functions to compute risk/edge metrics for trading signals.

Calculated values (all percentages expressed as decimals, e.g. 0.007 = 0.7%):
 - target_pct: относительное расстояние до TP
 - stop_pct: относительное расстояние до SL
 - cost_pct: оценка совокупных издержек (round-trip)
 - net_target_pct: target_pct - cost_pct
 - net_stop_pct: stop_pct + cost_pct
 - rrr: target_pct / stop_pct (Risk:Reward)
 - p_be: break-even вероятность попадания в TP до SL
 - ev_naive_pct: ожидание в процентах (используя p_dir ~ confidence как приближение p_hit)
 - edge_ok: проходит ли фильтр минимального чистого хода
 - prob_ok: проходит ли условие p_dir >= p_be

Config file (JSON): risk_config.json (опционально). Структура:
{
  "deposit": 8483,
  "risk_pct": 0.01,
  "leverage": 10,
  "default_round_trip_cost_pct": 0.002,
  "min_edge_pct": 0.005,
  "symbols": {
     "BTCUSDT": {"min_edge_pct": 0.007, "round_trip_cost_pct": 0.0022},
     "DOTUSDT": {"min_edge_pct": 0.005},
     "PEPEUSDT": {"min_edge_pct": 0.006, "round_trip_cost_pct": 0.0025}
  }
}

All fields optional; sensible fallbacks used if file or keys missing.
"""
from __future__ import annotations
import json
import os
from functools import lru_cache
from typing import Dict, Any

RISK_CONFIG_FILENAME = "risk_config.json"

DEFAULT_CONFIG = {
    "deposit": 10_000,               # используется только для информативных расчётов размера позиции (пока не выводим размер ордера)
    "risk_pct": 0.01,                # доля депозита, которую готовы рискнуть (собственный капитал)
    "leverage": 10,
    "default_round_trip_cost_pct": 0.003,  # 0.30% по умолчанию консервативно
    "min_edge_pct": 0.005,           # минимальный чистый ход (после комиссий) для принятия сделки
    "symbols": {}
}


@lru_cache(maxsize=1)
def load_risk_config(path: str | None = None) -> Dict[str, Any]:
    """Loads risk configuration from JSON file if present, else returns defaults.

    Caches result to avoid disk IO on each signal.
    """
    fname = path or RISK_CONFIG_FILENAME
    if os.path.isfile(fname):
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # merge shallow
            merged = DEFAULT_CONFIG.copy()
            merged.update({k: v for k, v in data.items() if k != 'symbols'})
            # symbols merge
            symbols = data.get('symbols', {})
            if not isinstance(symbols, dict):
                symbols = {}
            merged['symbols'] = symbols
            return merged
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def _get_symbol_overrides(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    return (cfg.get('symbols') or {}).get(symbol, {})


def compute_signal_metrics(signal: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compute risk / edge metrics for a raw signal dictionary.

    Required keys in signal: price (or close), stop_loss, take_profit, signal (LONG/SHORT), confidence.
    Returns dictionary with metrics. If insufficient data, returns {'error': '...'}.
    """
    if cfg is None:
        cfg = load_risk_config()  # cached

    price = signal.get('price') or signal.get('close')
    sl = signal.get('stop_loss')
    tp = signal.get('take_profit')
    side = signal.get('signal')
    confidence = signal.get('confidence')
    symbol = signal.get('symbol') or signal.get('symbol_name') or ''

    # Basic validation
    if side not in ('LONG', 'SHORT'):
        return {"warning": "Not a trading signal"}
    if price is None or sl is None or tp is None or confidence is None:
        return {"error": "Missing price/SL/TP/confidence"}
    try:
        price = float(price)
        sl = float(sl)
        tp = float(tp)
        confidence = float(confidence)
    except Exception:
        return {"error": "Non-numeric price/SL/TP/confidence"}
    if price <= 0:
        return {"error": "Invalid price"}

    # Distances
    if side == 'LONG':
        target_pct = (tp - price) / price
        stop_pct = (price - sl) / price
    else:  # SHORT
        target_pct = (price - tp) / price
        stop_pct = (sl - price) / price

    # Guard against negative or zero distances
    if target_pct <= 0 or stop_pct <= 0:
        return {"error": "Non-positive target/stop distances"}

    overrides = _get_symbol_overrides(cfg, symbol)
    cost_pct = overrides.get('round_trip_cost_pct', cfg.get('default_round_trip_cost_pct', 0.003))
    min_edge_pct = overrides.get('min_edge_pct', cfg.get('min_edge_pct', 0.005))
    min_net_rr = overrides.get('min_net_rr', cfg.get('min_net_rr', None))
    min_conf_edge_bp = overrides.get('min_conf_edge_bp', cfg.get('min_conf_edge_bp', None))  # в долях (0.005 = 0.5pp)
    reject_negative_cal_ev = overrides.get('reject_negative_calibrated_ev', cfg.get('reject_negative_calibrated_ev', False))

    net_target_pct = target_pct - cost_pct
    net_stop_pct = stop_pct + cost_pct

    # p_be (break-even) - avoid division by zero
    if net_target_pct <= 0:
        p_be = 1.0  # невозможно получить edge
    else:
        p_be = net_stop_pct / (net_stop_pct + net_target_pct)

    # Наивное EV в процентах от цены (используя confidence как proxy p_hit)
    ev_naive_pct = confidence * net_target_pct - (1 - confidence) * net_stop_pct

    rrr = target_pct / stop_pct if stop_pct > 0 else 0

    metrics = {
        'symbol': symbol,
        'side': side,
        'price': price,
        'target_pct': target_pct,
        'stop_pct': stop_pct,
        'rrr': rrr,
        'cost_pct': cost_pct,
        'net_target_pct': net_target_pct,
        'net_stop_pct': net_stop_pct,
        'p_be': p_be,
        'p_dir': confidence,
        'ev_naive_pct': ev_naive_pct,
        'min_edge_pct': min_edge_pct,
        'edge_ok': net_target_pct >= min_edge_pct,
        'prob_ok': confidence >= p_be,
        'net_rr': (net_target_pct / net_stop_pct) if net_stop_pct > 0 else 0,
        'min_net_rr': min_net_rr,
        'min_conf_edge_bp': min_conf_edge_bp,
        'reject_negative_calibrated_ev': reject_negative_cal_ev,
    }

    # === Position sizing (USD notionals) ===
    deposit = float(cfg.get('deposit', 0))
    risk_pct_cfg = float(cfg.get('risk_pct', 0))
    leverage = float(cfg.get('leverage', 1))
    sizing_mode = overrides.get('sizing_mode', cfg.get('sizing_mode', 'risk')).lower()
    # allow per symbol overrides if provided
    if overrides.get('risk_pct') is not None:
        risk_pct_cfg = float(overrides['risk_pct'])
    if overrides.get('leverage') is not None:
        leverage = float(overrides['leverage'])

    position_notional = None
    risk_amount_usd = None
    potential_profit_usd = None
    potential_loss_usd = None
    ev_naive_usd = None
    ev_calibrated_usd = None
    position_clamped = False

    if deposit > 0 and risk_pct_cfg > 0:
        if sizing_mode == 'risk':
            if net_stop_pct > 0:
                risk_amount_usd = deposit * risk_pct_cfg  # willing to lose
                try:
                    position_notional = risk_amount_usd / net_stop_pct
                except ZeroDivisionError:
                    position_notional = None
            else:
                risk_amount_usd = deposit * risk_pct_cfg
                position_notional = None
        else:  # margin mode
            # risk_pct трактуем как долю депозита, которую вносим как маржу
            margin_amount = deposit * risk_pct_cfg
            risk_amount_usd = None  # в margin режиме это не целевой риск
            position_notional = margin_amount * leverage
        if position_notional is not None:
            max_notional = deposit * leverage if leverage > 0 else deposit
            if position_notional > max_notional:
                position_notional = max_notional
                position_clamped = True
            potential_profit_usd = position_notional * net_target_pct
            potential_loss_usd = position_notional * net_stop_pct
            ev_naive_usd = position_notional * ev_naive_pct
    # (calibrated EV USD добавим позже если появится p_hit_cal)

    metrics.update({
        'deposit': deposit,
        'risk_pct_cfg': risk_pct_cfg,
        'leverage': leverage,
        'sizing_mode': sizing_mode,
        'position_notional': position_notional,
        'risk_amount_usd': risk_amount_usd,
        'potential_profit_usd': potential_profit_usd,
        'potential_loss_usd': potential_loss_usd,
        'ev_naive_usd': ev_naive_usd,
        'position_clamped': position_clamped,
    })

    # Опционально: если передан калиброванный p_hit_cal в signal, вычислим EV по нему
    p_hit_cal = signal.get('p_hit_cal')
    if isinstance(p_hit_cal, (int, float)) and 0 <= p_hit_cal <= 1:
        ev_calibrated_pct = p_hit_cal * net_target_pct - (1 - p_hit_cal) * net_stop_pct
        metrics['p_hit_cal'] = float(p_hit_cal)
        metrics['ev_calibrated_pct'] = ev_calibrated_pct
        metrics['ev_cal_flag'] = ev_calibrated_pct >= 0
        # Добавим ожидаемое значение в USD если есть позиция
        if metrics.get('position_notional') is not None:
            metrics['ev_calibrated_usd'] = metrics['position_notional'] * ev_calibrated_pct

    # Правила фильтра
    net_rr_ok = True if min_net_rr is None else (metrics['net_rr'] >= min_net_rr)
    conf_edge = metrics['p_dir'] - metrics['p_be']
    conf_edge_ok = True if min_conf_edge_bp is None else (conf_edge >= min_conf_edge_bp)
    cal_ev_ok = True
    if reject_negative_cal_ev and 'ev_calibrated_pct' in metrics:
        cal_ev_ok = metrics['ev_calibrated_pct'] >= 0
    metrics.update({
        'net_rr_ok': net_rr_ok,
        'conf_edge': conf_edge,
        'conf_edge_ok': conf_edge_ok,
        'cal_ev_ok': cal_ev_ok
    })
    return metrics


def format_metrics_block(m: Dict[str, Any]) -> str:
    """Produce a human-readable metrics block for Telegram."""
    if 'error' in m:
        return f"⚠ Metrics error: {m['error']}"
    if 'warning' in m:
        return ""
    def pct(x: float) -> str:
        return f"{x*100:.2f}%"
    flags = []
    if not m['edge_ok']:
        flags.append('EDGE')
    if not m['prob_ok']:
        flags.append('PROB')
    if m['ev_naive_pct'] < 0:
        flags.append('EV')
    if 'net_rr_ok' in m and not m['net_rr_ok']:
        flags.append('NET_RR')
    if 'conf_edge_ok' in m and not m['conf_edge_ok']:
        flags.append('CONF_EDGE')
    if 'cal_ev_ok' in m and not m['cal_ev_ok']:
        flags.append('CAL_EV')
    flag_text = (" ⚠ (" + ",".join(flags) + ")") if flags else ""
    block = (
        "\n\n💹 <b>Risk Metrics</b>:\n"
    f"Target: {pct(m['target_pct'])} \nStop: {pct(m['stop_pct'])} \nR:R={m['rrr']:.2f} \nNetR:R={m.get('net_rr',0):.2f}\n"
        f"NetTarget: {pct(m['net_target_pct'])} \nNetStop: {pct(m['net_stop_pct'])}\n"
        f"Cost: {pct(m['cost_pct'])} \np_be: {m['p_be']*100:.1f}% \np_dir: {m['p_dir']*100:.1f}%\n"
        f"EV(naive): {pct(m['ev_naive_pct'])}{flag_text}"
    )
    if 'ev_calibrated_pct' in m and 'p_hit_cal' in m:
        block += f"\nEV(cal): {pct(m['ev_calibrated_pct'])} (p_hit={m['p_hit_cal']*100:.1f}%)"
    # Position sizing / P&L block
    if m.get('position_notional'):
        def usd(x):
            return f"{x:,.2f}" if x is not None else "-"
        pos_line = f"\n💼 Pos: {usd(m['position_notional'])} USDT ({m.get('sizing_mode','risk')})"
        if m.get('sizing_mode') == 'risk' and m.get('risk_amount_usd') is not None:
            pos_line += f"\nRiskAmt: {usd(m['risk_amount_usd'])}"
        elif m.get('sizing_mode') == 'margin':
            # восстановим маржу: position_notional / leverage
            lev = m.get('leverage', 1) or 1
            margin_est = (m['position_notional'] / lev) if lev else None
            if margin_est:
                pos_line += f" | Margin: {usd(margin_est)}"
        if m.get('position_clamped'):
            pos_line += " (clamped)"
        pnl_line = ""
        if m.get('potential_profit_usd') is not None and m.get('potential_loss_usd') is not None:
            pnl_line = (f"\nP&L TP/SL: +{usd(m['potential_profit_usd'])} / -{usd(m['potential_loss_usd'])}")
        ev_lines = ""
        if m.get('ev_naive_usd') is not None:
            ev_lines += f"\nEV(naive): {usd(m['ev_naive_usd'])} USDT"
        if m.get('ev_calibrated_usd') is not None:
            ev_lines += f"\nEV(cal): {usd(m['ev_calibrated_usd'])} USDT"
        block += pos_line + pnl_line + ev_lines
    return block


__all__ = [
    'load_risk_config',
    'compute_signal_metrics',
    'format_metrics_block'
]
