#!/usr/bin/env python3
"""Generate Markdown comparison table for RL models by symbol.

- Scans `rl_system/models/*`.
- For each model directory, reads root `config.json` and `selected_best_by_metrics.json`.
- Filters models by `symbol` (e.g. ADAUSDT, BTCUSDT).
- Outputs a Markdown table with metrics of the best checkpoint chosen by selector.

Usage examples:

    python rl_system/generate_models_md_table.py --symbol ADAUSDT
    python rl_system/generate_models_md_table.py --symbol BTCUSDT --output RL_BTCUSDT_MODELS.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compute_stats_from_checkpoints_md(model_dir: Path) -> Dict[str, Any]:
    """Fallback: compute checkpoint stats from CHECKPOINTS_COMPARISON.md table.
    
    Used for existing models where checkpoint_stats is not yet in the JSON.
    Parses the "All Checkpoints Performance" table and the Plateau/Verdict sections.
    """
    md_path = model_dir / "CHECKPOINTS_COMPARISON.md"
    if not md_path.exists():
        return {}
    
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    
    # Parse checkpoint table rows
    lines = text.split("\n")
    returns = []
    pfs = []
    in_table = False
    for line in lines:
        if "| Step |" in line and "Return" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 4:
                try:
                    step_str = cols[0].replace(",", "")
                    ret_str = cols[2].replace("+", "").replace("%", "").strip()
                    pf_str = cols[3].strip()
                    step = int(step_str) if step_str.isdigit() else 0
                    ret_val = float(ret_str) if ret_str and ret_str != "-" else None
                    pf_val = float(pf_str) if pf_str and pf_str != "-" else None
                    # Skip special models (best/final with huge step numbers)
                    if step < 900_000_000 and ret_val is not None:
                        returns.append(ret_val)
                        pfs.append(pf_val)
                except (ValueError, IndexError):
                    pass
        elif in_table and not line.startswith("|"):
            in_table = False
    
    if not returns:
        return {}
    
    total = len(returns)
    positive = sum(1 for r in returns if r > 0)
    positive_pct = round(positive * 100.0 / total, 1) if total > 0 else 0.0
    
    # Tail profitable (last 20%)
    tail_size = max(1, total // 5)
    tail_rets = returns[-tail_size:]
    tail_pfs = pfs[-tail_size:]
    tail_profitable = sum(
        1 for r, p in zip(tail_rets, tail_pfs)
        if r is not None and r > 0 and p is not None and p >= 1.0
    )
    tail_profitable_pct = round(tail_profitable * 100.0 / len(tail_rets), 1) if tail_rets else 0.0
    
    # Parse plateau info from markdown 
    plateau_found = False
    plateau_len = 0
    
    # Look for "✅ **Stable plateau found" or "❌ **No stable plateau"
    plateau_match = re.search(r"✅\s*\*\*Stable plateau found.*?(\d+)\s*consecutive", text)
    if plateau_match:
        plateau_found = True
        plateau_len = int(plateau_match.group(1))
    else:
        no_plateau_match = re.search(r"longest region.*?is\s*(\d+)\s*checkpoints", text)
        if no_plateau_match:
            plateau_len = int(no_plateau_match.group(1))
    
    # Parse verdict
    verdict = None
    if "## Live Readiness Verdict" in text:
        if re.search(r"✅\s*\*\*PASS", text):
            verdict = "PASS"
        elif re.search(r"❌\s*\*\*FAIL", text):
            verdict = "FAIL"
    
    return {
        "total_checkpoints": total,
        "valid_checkpoints": total,
        "positive_pct": positive_pct,
        "plateau_found": plateau_found,
        "plateau_len": plateau_len,
        "tail_profitable_pct": tail_profitable_pct,
        "verdict": verdict,
        "verdict_reasons": [],
    }


def load_model_entry(model_dir: Path, target_symbol: str) -> Optional[Dict[str, Any]]:
    """Load summary entry for a single model directory.

    Expects:
    - root `config.json` with fields: symbol, timeframe, algorithm, days, total_timesteps
    - `selected_best_by_metrics.json` with metrics of the best checkpoint

    Returns None if symbol does not match or files are missing/invalid.
    """
    root_config_path = model_dir / "config.json"
    selector_path = model_dir / "selected_best_by_metrics.json"

    if not root_config_path.exists() or not selector_path.exists():
        return None

    try:
        with root_config_path.open("r", encoding="utf-8") as f:
            root_cfg = json.load(f)
    except Exception:
        return None

    if root_cfg.get("symbol") != target_symbol:
        return None

    try:
        with selector_path.open("r", encoding="utf-8") as f:
            sel = json.load(f)
    except Exception:
        return None

    # Поддержка нового формата (с best_checkpoint) и старого (метрики на верхнем уровне)
    if "best_checkpoint" in sel:
        # Новый формат
        metrics = sel["best_checkpoint"]
    else:
        # Старый формат (обратная совместимость)
        metrics = sel

    # Basic model info from root config
    symbol = root_cfg.get("symbol", "?")
    timeframe = root_cfg.get("timeframe", "?")
    algorithm = root_cfg.get("algorithm", "?")
    days = root_cfg.get("days", "?")
    total_timesteps = root_cfg.get("total_timesteps", 0)
    config_file = root_cfg.get("config_file", "")

    # Best-checkpoint metrics from selector
    ret = metrics.get("total_return_pct")
    pf = metrics.get("profit_factor")
    trades = metrics.get("total_trades")
    win_rate = metrics.get("win_rate_pct")
    max_dd = metrics.get("max_drawdown_pct")
    episode_reward = metrics.get("episode_reward")
    episodes = metrics.get("episodes")

    # Try to extract checkpoint steps from model_path (e.g. rl_model_240000_steps.zip)
    checkpoint_label = ""
    model_path = metrics.get("model_path", "")
    if isinstance(model_path, str) and model_path:
        stem = Path(model_path).stem  # rl_model_240000_steps
        # Simple heuristic: keep stem as-is; it's already informative
        checkpoint_label = stem

    # Derive some notes (very lightweight, optional)
    notes: List[str] = []
    if isinstance(trades, (int, float)) and trades is not None:
        if trades < 20:
            notes.append("few trades")
        elif trades > 250:
            notes.append("high trading frequency")
    if isinstance(pf, (int, float)) and pf is not None:
        if pf >= 1.5:
            notes.append("good PF")
        elif pf < 1.0:
            notes.append("unprofitable")
    if isinstance(ret, (int, float)) and ret is not None:
        if ret > 30:
            notes.append("high return")
        elif ret < 0:
            notes.append("negative return")

    notes_str = "; ".join(notes) if notes else ""
    
    # Извлекаем только имя файла конфига без пути и расширения
    config_file_name = ""
    if config_file:
        config_file_name = Path(config_file).stem  # Убираем путь и .json

    # Checkpoint stats (новые поля для оценки надёжности)
    ckpt_stats = sel.get("checkpoint_stats", {})
    # Fallback: если checkpoint_stats нет в JSON, парсим из CHECKPOINTS_COMPARISON.md
    if not ckpt_stats:
        ckpt_stats = _compute_stats_from_checkpoints_md(model_dir)
    positive_pct = ckpt_stats.get("positive_pct")
    plateau_found = ckpt_stats.get("plateau_found")
    plateau_len = ckpt_stats.get("plateau_len")
    tail_profitable_pct = ckpt_stats.get("tail_profitable_pct")
    verdict = ckpt_stats.get("verdict")

    # Дополнительные предупреждения в Notes на основе checkpoint_stats
    if isinstance(episodes, (int, float)) and episodes is not None and episodes < 10:
        if not plateau_found:
            notes.append("early ckpt risk")
    if isinstance(positive_pct, (int, float)) and positive_pct < 25:
        notes.append("low pos%")
    notes_str = "; ".join(notes) if notes else ""

    return {
        "model_dir": model_dir.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "algorithm": algorithm,
        "days": days,
        "timesteps": int(total_timesteps) if isinstance(total_timesteps, (int, float)) else total_timesteps,
        "checkpoint": checkpoint_label,
        "config_file": config_file_name,
        "return_pct": ret,
        "profit_factor": pf,
        "trades": trades,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd,
        "episode_reward": episode_reward,
        "episodes": episodes,
        "positive_pct": positive_pct,
        "plateau_found": plateau_found,
        "plateau_len": plateau_len,
        "tail_profitable_pct": tail_profitable_pct,
        "verdict": verdict,
        "notes": notes_str,
    }


def build_markdown(symbol: str, entries: List[Dict[str, Any]]) -> str:
    """Build Markdown table content for given entries."""
    header = f"# {symbol} RL Models (Best Checkpoints)\n\n"

    if not entries:
        return header + "_No models found for this symbol._\n"

    # Sort by return DESC, then by PF DESC
    entries_sorted = sorted(
        entries,
        key=lambda e: (
            e["return_pct"] if isinstance(e["return_pct"], (int, float)) else -1e9,
            e["profit_factor"] if isinstance(e["profit_factor"], (int, float)) else -1e9,
        ),
        reverse=True,
    )

    lines: List[str] = []
    # Column order: Model, Best checkpoint (for quick copy-paste), then meta/metrics
    lines.append("| Model | Best Ckpt | Config | Algo | TF | Days | Steps | Episode | Return % | PF | Trades | Win % | Max DD % | Pos% | Plateau | Verdict | Notes |")
    lines.append("|-------|-----------|--------|------|----|------|-------|---------|----------|----|--------|-------|----------|------|---------|---------|-------|")

    def fmt(x: Any, digits: int = 2) -> str:
        if isinstance(x, (int, float)):
            return f"{x:.{digits}f}" if isinstance(x, float) else str(x)
        return "-"

    for e in entries_sorted:
        # Format episodes: show as integer if whole, 1 decimal otherwise, "-" if missing
        ep_val = e.get("episodes")
        if isinstance(ep_val, (int, float)) and ep_val is not None:
            ep_str = str(int(ep_val)) if ep_val == int(ep_val) else f"{ep_val:.1f}"
        else:
            ep_str = "-"
        
        # Format positive_pct
        pos_pct = e.get("positive_pct")
        pos_str = f"{pos_pct:.0f}%" if isinstance(pos_pct, (int, float)) else "-"
        
        # Format plateau
        p_found = e.get("plateau_found")
        p_len = e.get("plateau_len")
        if p_found is True and isinstance(p_len, (int, float)):
            plat_str = str(int(p_len))
        elif p_found is False:
            plat_str = "\u2014"
        else:
            plat_str = "-"
        
        # Format verdict
        v = e.get("verdict")
        if v == "PASS":
            verdict_str = "\u2705"
        elif v == "FAIL":
            verdict_str = "\u274c"
        else:
            verdict_str = "-"
        
        lines.append(
            "| {model} | {ckpt} | {config} | {algo} | {tf} | {days} | {steps} | {episode} | {ret} | {pf} | {trades} | {win} | {dd} | {pos} | {plat} | {verdict} | {notes} |".format(
                model=e["model_dir"],
                ckpt=e["checkpoint"] or "-",
                config=e.get("config_file", "-") or "-",
                algo=e["algorithm"],
                tf=e["timeframe"],
                days=e["days"],
                steps=e["timesteps"],
                episode=ep_str,
                ret=fmt(e["return_pct"]),
                pf=fmt(e["profit_factor"]),
                trades=e["trades"] if e["trades"] is not None else "-",
                win=fmt(e["win_rate_pct"]),
                dd=fmt(e["max_drawdown_pct"]),
                pos=pos_str,
                plat=plat_str,
                verdict=verdict_str,
                notes=e["notes"],
            )
        )

    return header + "\n".join(lines) + "\n"


def main() -> None:
    # Настройка вывода для Windows консоли
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(description="Generate Markdown table for RL models of a given symbol.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ADAUSDT or BTCUSDT")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .md file path (default: RL_<SYMBOL>_MODELS_TABLE.md in project root)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()

    models_root = Path("rl_system/models")
    if not models_root.exists():
        print(f"❌ Models directory not found: {models_root}")
        return

    entries: List[Dict[str, Any]] = []
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue
        entry = load_model_entry(model_dir, symbol)
        if entry:
            entries.append(entry)

    md = build_markdown(symbol, entries)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"RL_{symbol}_MODELS_TABLE.md")

    out_path.write_text(md, encoding="utf-8")
    print(f"✅ Wrote Markdown table for {symbol} to: {out_path}")
    print(f"   Models found: {len(entries)}")


if __name__ == "__main__":
    main()
