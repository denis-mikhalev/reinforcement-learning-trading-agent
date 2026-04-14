import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "rl_system" / "best_models_summary.json"
TXT_REGISTRY_PATH = ROOT / "rl_system" / "live_models_registry.txt"


def load_registry():
    """Загружает реестр моделей для live‑запуска.

    Приоритет форматов:
      1) Простой текстовый файл live_models_registry.txt
         Формат строк:
             MODEL_FOLDER CHECKPOINT_NAME
         например:
             ADAUSDT_4h_A2C_1300d_bt60d_20251212_173627 rl_model_280000_steps
             ADAUSDT_6h_A2C_800d_bt60d_20251209_122423 rl_model_160000_steps

         MODEL_FOLDER интерпретируется как поддиректория в rl_system/models/.
         CHECKPOINT_NAME интерпретируется как файл внутри checkpoints/.

      2) JSON best_models_summary.json (старый формат, как сейчас).
    """

    # 1. Новый простой текстовый формат
    if TXT_REGISTRY_PATH.exists():
        models = []
        with open(TXT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                folder = parts[0]
                checkpoint = parts[1] if len(parts) > 1 else None

                if checkpoint:
                    # Если пользователь не указал .zip, добавляем его автоматически
                    if not checkpoint.endswith(".zip"):
                        checkpoint_file = checkpoint + ".zip"
                    else:
                        checkpoint_file = checkpoint

                    model_rel_path = Path("rl_system") / "models" / folder / "checkpoints" / checkpoint_file
                else:
                    # Если чекпоинт не указан, передаём директорию модели —
                    # live_agent сам найдёт подходящий .zip внутри
                    model_rel_path = Path("rl_system") / "models" / folder

                models.append({
                    "label": folder,
                    "description": "",
                    "model_path": str(model_rel_path).replace("\\", "/"),
                    "enabled_live": True,
                })

        return {"models": models}

    # 2. Старый JSON‑формат (best_models_summary.json)
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_live_state_dir():
    """Удаляет старые файлы состояния из rl_system/live_state перед новой live‑сессией.

    Это позволяет избежать ситуации, когда ensemble‑сводка показывает
    сигналы от давно остановленных моделей.
    """

    live_state_dir = ROOT / "rl_system" / "live_state"
    if not live_state_dir.exists():
        return

    removed = 0
    for path in live_state_dir.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue

    if removed:
        print(f"🧹 Cleared {removed} stale live_state file(s) before live launch.")


def select_models(registry, labels=None):
    models = registry.get("models", [])

    if labels:
        wanted = set(labels)
        return [m for m in models if m.get("label") in wanted]

    # default: take all with enabled_live == true
    return [m for m in models if m.get("enabled_live")]


def build_command(model_path: str, continuous: bool, telegram: bool, extra_args=None):
    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"

    # Внутренняя команда, которую нужно выполнить в новом окне
    # Без внешних кавычек вокруг всей строки, чтобы cmd.exe корректно распознал путь
    base_cmd = [
        str(python_exe),
        "rl_system/run_live_agent.py",
        "--model-path",
        model_path,
    ]
    if continuous:
        base_cmd.append("--continuous")
    if telegram:
        base_cmd.append("--telegram")
    if extra_args:
        base_cmd.extend(extra_args)

    # Join into a single command string for cmd.exe
    inner = " ".join(base_cmd)

    # On Windows, spawn a separate cmd.exe window
    if sys.platform.startswith("win"):
        # start "" cmd /k <command>
        return ["cmd.exe", "/c", "start", "" , "cmd", "/k", inner]

    # Fallback for non-Windows (still useful if ever run elsewhere)
    return ["bash", "-lc", inner]


def build_ensemble_command(interval: int = 60, telegram: bool = True):
    """Build command to launch ensemble live summary in a separate terminal window."""

    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"

    base_cmd = [
        str(python_exe),
        "rl_system/live_signals_summary.py",
        "--interval",
        str(interval),
    ]
    if telegram:
        base_cmd.append("--telegram")

    inner = " ".join(base_cmd)

    if sys.platform.startswith("win"):
        return ["cmd.exe", "/c", "start", "", "cmd", "/k", inner]

    return ["bash", "-lc", inner]


def main():
    parser = argparse.ArgumentParser(description="Launch RL live agents from best_models_summary.json")
    parser.add_argument("--label", action="append", help="Launch only specified model label(s). If omitted, uses enabled_live flags.")
    parser.add_argument("--single-shot", action="store_true", help="Run agents in single-shot mode (no --continuous)")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram even if normally used")
    parser.add_argument("--dry-run", action="store_true", help="Only print which commands would be run")

    # Extra args that control sound behaviour for underlying live agents.
    # These are passed through to rl_system/run_live_agent.py:
    #   --no-sound             -> completely mute
    #   --signal-change-sound  -> beep only when signal changes
    parser.add_argument("--no-sound", action="store_true", help="Run agents with no sound (passed through)")
    parser.add_argument("--signal-change-sound", action="store_true", help="Beep only when signal changes (passed through)")

    args, extra = parser.parse_known_args()

    registry = load_registry()
    selected = select_models(registry, labels=args.label)

    if not selected:
        print("No models selected for live launch. Check labels or enabled_live flags.")
        return

    # Перед новой live‑сессией очищаем старые файлы состояния,
    # чтобы ensemble‑сводка отражала только актуально запущенные модели.
    if not args.dry_run:
        clear_live_state_dir()

    # If called with --signal-change-sound (your main live task),
    # automatically start the ensemble summary monitor in a separate cmd window.
    if args.signal_change_sound and not args.dry_run:
        try:
            ensemble_cmd = build_ensemble_command(interval=60, telegram=True)
            print(f"Starting RL ensemble summary: {' '.join(ensemble_cmd)}")
            subprocess.Popen(ensemble_cmd, cwd=str(ROOT))
        except Exception as e:
            print(f"⚠️ Could not start ensemble summary: {e}")

    print("Launching RL live agents for models:")
    for m in selected:
        label = m.get("label")
        model_path = m.get("model_path")
        continuous = not args.single_shot
        telegram = not args.no_telegram

        # Compose extra arguments for the live agent process.
        extra_args = list(extra) if extra else []
        if args.no_sound and "--no-sound" not in extra_args:
            extra_args.append("--no-sound")
        if args.signal_change_sound and "--signal-change-sound" not in extra_args:
            extra_args.append("--signal-change-sound")

        cmd = build_command(model_path, continuous=continuous, telegram=telegram, extra_args=extra_args)
        print(f" - {label}: {' '.join(cmd)}")

        if args.dry_run:
            continue

        # run each agent in its own process / terminal window
        subprocess.Popen(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
