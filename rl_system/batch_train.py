"""
Batch Training Script
=====================
Последовательное обучение нескольких моделей RL из списка конфигураций.

Использование:
    python rl_system/batch_train.py --config-list rl_system/training_queue.txt
    python rl_system/batch_train.py --config-list rl_system/training_queue.txt --continue-on-error
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


def log_message(message: str, file=None):
    """Логирование с timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    if file:
        file.write(log_line + "\n")
        file.flush()


def train_model(config_path: str, python_exe: str, log_file) -> bool:
    """
    Запускает обучение одной модели
    
    Returns:
        bool: True если обучение успешно, False если ошибка
    """
    log_message(f"=== Starting training: {config_path} ===", log_file)
    
    cmd = [
        python_exe,
        "-u",  # Unbuffered output для реального времени
        "rl_system/train_agent_v2.py",
        "--config",
        config_path
    ]
    
    log_message(f"Command: {' '.join(cmd)}", log_file)
    
    try:
        # Настройка окружения для правильной кодировки UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # Запускаем процесс БЕЗ перенаправления stdout
        # Это позволяет прогресс-барам и эмоджи работать корректно
        process = subprocess.Popen(
            cmd,
            env=env,
            bufsize=0  # Unbuffered
        )
        
        # Ждем завершения
        return_code = process.wait()
        
        if return_code == 0:
            log_message(f"✓ Successfully completed: {config_path}", log_file)
            return True
        else:
            log_message(f"✗ Failed with return code {return_code}: {config_path}", log_file)
            return False
            
    except Exception as e:
        log_message(f"✗ Exception during training: {str(e)}", log_file)
        return False


def load_config_list(file_path: str) -> list:
    """
    Загружает список конфигов из файла
    
    Формат файла:
    - Один путь к конфигу на строку
    - Строки начинающиеся с # игнорируются (комментарии)
    - Пустые строки игнорируются
    """
    configs = []
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return configs
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Пропускаем комментарии и пустые строки
            if not line or line.startswith('#'):
                continue
            
            # Проверяем что конфиг существует
            if not Path(line).exists():
                print(f"Warning: Config not found (line {line_num}): {line}")
                continue
            
            configs.append(line)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Batch training for RL models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-list",
        type=str,
        required=True,
        help="Path to file with list of configs (one per line)"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training next models even if one fails"
    )
    
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Path to Python executable"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional, will be auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Загружаем список конфигов
    configs = load_config_list(args.config_list)
    
    if not configs:
        print(f"Error: No valid configs found in {args.config_list}")
        return 1
    
    # Создаем лог-файл
    if args.log_file:
        log_path = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"rl_system/logs/batch_train_{timestamp}.log"
    
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Основной цикл обучения
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_message("=" * 70, log_file)
        log_message("BATCH TRAINING STARTED", log_file)
        log_message(f"Config list: {args.config_list}", log_file)
        log_message(f"Total configs: {len(configs)}", log_file)
        log_message(f"Continue on error: {args.continue_on_error}", log_file)
        log_message(f"Log file: {log_path}", log_file)
        log_message("=" * 70, log_file)
        log_message("", log_file)
        
        results = []
        
        for idx, config_path in enumerate(configs, 1):
            log_message("", log_file)
            log_message("=" * 70, log_file)
            log_message(f"Training model {idx}/{len(configs)}", log_file)
            log_message("=" * 70, log_file)
            
            success = train_model(config_path, args.python_exe, log_file)
            results.append({
                'config': config_path,
                'success': success
            })
            
            if not success and not args.continue_on_error:
                log_message("", log_file)
                log_message("Training failed and --continue-on-error not set. Stopping.", log_file)
                break
        
        # Итоговая статистика
        log_message("", log_file)
        log_message("=" * 70, log_file)
        log_message("BATCH TRAINING COMPLETED", log_file)
        log_message("=" * 70, log_file)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        log_message(f"Total trained: {len(results)}/{len(configs)}", log_file)
        log_message(f"Successful: {successful}", log_file)
        log_message(f"Failed: {failed}", log_file)
        log_message("", log_file)
        
        if failed > 0:
            log_message("Failed configs:", log_file)
            for r in results:
                if not r['success']:
                    log_message(f"  - {r['config']}", log_file)
        
        log_message("", log_file)
        log_message(f"Full log saved to: {log_path}", log_file)
        log_message("=" * 70, log_file)
    
    # Возвращаем код ошибки если были неудачи
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
