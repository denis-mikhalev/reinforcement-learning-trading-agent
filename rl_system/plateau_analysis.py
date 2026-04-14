"""Plateau / stability analysis for checkpoint backtests.

This module is intentionally lightweight and self-contained so it can be used both
from training (`train_agent_v2.py`) and from post-hoc regeneration
(`regenerate_checkpoint_visualization.py`).

The goal is not to "prove" tradability, but to provide a practical warning when
performance looks like an isolated spike rather than a stable region.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SPECIAL_TIMESTEP_CUTOFF = 900_000_000


@dataclass(frozen=True)
class PlateauThresholds:
    # Basic validity / trading sanity
    min_trades: int = 30
    pf_min: float = 1.30  # Повышено с 1.20 для более строгого отбора
    dd_max: float = 25.0  # Снижено с 30.0 для меньшего риска
    min_return_pct: float = 5.0  # Повышено с 0.0 для отсечения слабых моделей

    # Plateau definition (based on score proximity to the best score)
    plateau_min_len: int = 3  # Снижено с 5 для более реалистичного обнаружения плато
    score_eps_abs: float = 5.0  # Повышено с 2.0 для большего допуска на вариативность
    score_eps_rel: float = 0.20  # Повышено с 0.15 для учета естественной волатильности

    # Tail stability (recent checkpoints)
    tail_len: int = 5  # Снижено с 10 для фокуса на последних чекпоинтах
    tail_min_profitable_ratio: float = 0.6  # Повышено с 0.4 для более стабильной концовки


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def load_thresholds_from_config(model_dir: Path) -> PlateauThresholds:
    """Loads optional thresholds from model's config.json.

    Backward compatible: if fields are missing, defaults are used.

    Supported fields (top-level in config.json):
    - live_min_trades or min_trades (int, default: 30)
    - live_pf_min (float, default: 1.30)
    - live_dd_max (float, default: 25.0)
    - live_min_return_pct (float, default: 5.0)
    - plateau_min_len (int, default: 3)
    - plateau_score_eps_abs (float, default: 5.0)
    - plateau_score_eps_rel (float, default: 0.20)
    - tail_len or live_tail_len (int, default: 5)
    - tail_min_profitable_ratio or live_tail_min_profitable_ratio (float, default: 0.6)
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return PlateauThresholds()

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return PlateauThresholds()

    return PlateauThresholds(
        min_trades=_safe_int(cfg.get("live_min_trades", cfg.get("min_trades", PlateauThresholds.min_trades)), PlateauThresholds.min_trades),
        pf_min=_safe_float(cfg.get("live_pf_min", PlateauThresholds.pf_min), PlateauThresholds.pf_min),
        dd_max=_safe_float(cfg.get("live_dd_max", PlateauThresholds.dd_max), PlateauThresholds.dd_max),
        min_return_pct=_safe_float(cfg.get("live_min_return_pct", PlateauThresholds.min_return_pct), PlateauThresholds.min_return_pct),
        plateau_min_len=_safe_int(cfg.get("plateau_min_len", PlateauThresholds.plateau_min_len), PlateauThresholds.plateau_min_len),
        score_eps_abs=_safe_float(cfg.get("plateau_score_eps_abs", PlateauThresholds.score_eps_abs), PlateauThresholds.score_eps_abs),
        score_eps_rel=_safe_float(cfg.get("plateau_score_eps_rel", PlateauThresholds.score_eps_rel), PlateauThresholds.score_eps_rel),
        tail_len=_safe_int(cfg.get("tail_len", cfg.get("live_tail_len", PlateauThresholds.tail_len)), PlateauThresholds.tail_len),
        tail_min_profitable_ratio=_safe_float(
            cfg.get("tail_min_profitable_ratio", cfg.get("live_tail_min_profitable_ratio", PlateauThresholds.tail_min_profitable_ratio)),
            PlateauThresholds.tail_min_profitable_ratio,
        ),
    )


def _is_regular_checkpoint(m: dict) -> bool:
    return _safe_int(m.get("timestep", 0)) < _SPECIAL_TIMESTEP_CUTOFF


def _is_valid_for_plateau(m: dict, thresholds: PlateauThresholds) -> bool:
    # Keep this minimal and aligned with your scoring gate.
    trades_ok = _safe_int(m.get("total_trades", 0)) >= thresholds.min_trades
    episodes_ok = _safe_float(m.get("episodes", 0.0)) >= 2.0
    return trades_ok and episodes_ok


def _is_profitable(m: dict) -> bool:
    # "Profitable" here is a soft notion: positive return and PF >= 1.
    return _safe_float(m.get("total_return_pct", 0.0)) > 0.0 and _safe_float(m.get("profit_factor", 0.0)) >= 1.0


def compute_plateau(metrics_list: list[dict], best_checkpoint_name: str, thresholds: PlateauThresholds) -> dict:
    """Computes plateau statistics based on consecutive checkpoints close to the best score."""
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    regular = sorted(regular, key=lambda x: _safe_int(x.get("timestep", 0)))

    if not regular:
        return {
            "found": False,
            "reason": "no_regular_checkpoints",
        }

    best = next((m for m in regular if best_checkpoint_name in str(m.get("checkpoint", ""))), None)
    if best is None:
        # Fallback: best by score among valid
        valid = [m for m in regular if _is_valid_for_plateau(m, thresholds)]
        best = max(valid, key=lambda x: _safe_float(x.get("score", -1e9)), default=None)

    if best is None:
        return {
            "found": False,
            "reason": "no_valid_checkpoints_for_plateau",
        }

    best_score = _safe_float(best.get("score", 0.0))
    eps_used = max(thresholds.score_eps_abs, abs(best_score) * thresholds.score_eps_rel)

    candidates = [m for m in regular if _is_valid_for_plateau(m, thresholds)]
    if not candidates:
        return {
            "found": False,
            "reason": "no_valid_checkpoints_for_plateau",
            "best_score": best_score,
            "eps_used": eps_used,
        }

    def in_plateau(m: dict) -> bool:
        return _safe_float(m.get("score", -1e9)) >= best_score - eps_used

    # Find the longest consecutive segment.
    longest = []
    current = []
    for m in candidates:
        if in_plateau(m):
            current.append(m)
        else:
            if len(current) > len(longest):
                longest = current
            current = []
    if len(current) > len(longest):
        longest = current

    found = len(longest) >= thresholds.plateau_min_len

    plateau_steps = [
        {
            "checkpoint": m.get("checkpoint", ""),
            "timestep": _safe_int(m.get("timestep", 0)),
            "score": _safe_float(m.get("score", 0.0)),
            "return_pct": _safe_float(m.get("total_return_pct", 0.0)),
            "pf": _safe_float(m.get("profit_factor", 0.0)),
            "trades": _safe_int(m.get("total_trades", 0)),
            "dd": _safe_float(m.get("max_drawdown_pct", 0.0)),
        }
        for m in longest
    ]

    return {
        "found": found,
        "best_score": best_score,
        "eps_used": eps_used,
        "min_len": thresholds.plateau_min_len,
        "len": len(longest),
        "start_step": plateau_steps[0]["timestep"] if plateau_steps else None,
        "end_step": plateau_steps[-1]["timestep"] if plateau_steps else None,
        "steps": plateau_steps,
        "note": "Plateau is measured on OOS checkpoint backtests using score proximity to the best checkpoint.",
    }


def compute_live_verdict(metrics_list: list[dict], best_checkpoint_name: str, thresholds: PlateauThresholds, plateau: dict) -> dict:
    """Builds a simple PASS/FAIL verdict with human-readable reasons."""
    regular = [m for m in metrics_list if _is_regular_checkpoint(m)]
    regular = sorted(regular, key=lambda x: _safe_int(x.get("timestep", 0)))

    best = next((m for m in regular if best_checkpoint_name in str(m.get("checkpoint", ""))), None)
    if best is None and regular:
        best = max(regular, key=lambda x: _safe_float(x.get("score", -1e9)))

    reasons: list[str] = []

    # Plateau gate
    if not plateau or not plateau.get("found", False):
        plateau_len = _safe_int(plateau.get("len", 0), 0) if isinstance(plateau, dict) else 0
        if plateau_len == 0:
            reasons.append("No plateau detected (performance looks like isolated spikes)")
        else:
            reasons.append(f"Plateau too short: {plateau_len} checkpoints (min {thresholds.plateau_min_len})")

    # Best checkpoint sanity
    if best is not None:
        trades = _safe_int(best.get("total_trades", 0))
        pf = _safe_float(best.get("profit_factor", 0.0))
        dd = _safe_float(best.get("max_drawdown_pct", 0.0))
        ret = _safe_float(best.get("total_return_pct", 0.0))

        if trades < thresholds.min_trades:
            reasons.append(f"Best checkpoint has too few trades: {trades} (min {thresholds.min_trades})")
        if pf < thresholds.pf_min:
            reasons.append(f"Best checkpoint PF below threshold: {pf:.2f} (min {thresholds.pf_min:.2f})")
        if dd > thresholds.dd_max:
            reasons.append(f"Best checkpoint drawdown too high: {dd:.2f}% (max {thresholds.dd_max:.2f}%)")
        if ret < thresholds.min_return_pct:
            reasons.append(
                f"Best checkpoint return below threshold: {ret:+.2f}% (min {thresholds.min_return_pct:+.2f}%)"
            )

    # Tail stability check (recent checkpoints)
    tail = [m for m in regular if _is_valid_for_plateau(m, thresholds)]
    if tail:
        tail_slice = tail[-max(1, thresholds.tail_len):]
        profitable_ratio = sum(1 for m in tail_slice if _is_profitable(m)) / len(tail_slice)
        if profitable_ratio < thresholds.tail_min_profitable_ratio:
            reasons.append(
                f"Recent checkpoints are unstable: profitable ratio in last {len(tail_slice)} is {profitable_ratio:.2f} "
                f"(min {thresholds.tail_min_profitable_ratio:.2f})"
            )

    status = "PASS" if len(reasons) == 0 else "FAIL"
    return {
        "status": status,
        "reasons": reasons,
        "thresholds": {
            "min_trades": thresholds.min_trades,
            "pf_min": thresholds.pf_min,
            "dd_max": thresholds.dd_max,
            "min_return_pct": thresholds.min_return_pct,
            "plateau_min_len": thresholds.plateau_min_len,
            "score_eps_abs": thresholds.score_eps_abs,
            "score_eps_rel": thresholds.score_eps_rel,
            "tail_len": thresholds.tail_len,
            "tail_min_profitable_ratio": thresholds.tail_min_profitable_ratio,
        },
    }
