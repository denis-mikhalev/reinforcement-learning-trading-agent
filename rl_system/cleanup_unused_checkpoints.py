"""
Скрипт для удаления неиспользуемых чекпоинтов моделей RL.

Оставляет только лучший чекпоинт для каждой модели:
- Если модель указана в live_models_registry.txt - использует чекпоинт из registry
- Если модель не указана в registry - использует чекпоинт из selected_best_by_metrics.json
- Удаляет все остальные чекпоинты, а также best_model.zip и final_model.zip

Использование:
    python cleanup_unused_checkpoints.py [--dry-run] [--yes]

Опции:
    --dry-run    Показать, что будет удалено, но не удалять
    --yes        Не спрашивать подтверждения
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Set, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "rl_system" / "live_models_registry.txt"
MODELS_DIR = ROOT / "rl_system" / "models"


def load_used_checkpoints() -> Dict[str, str]:
    """Загружает список используемых чекпоинтов из registry.
    
    Returns:
        Dict[model_folder, checkpoint_name]: Словарь папок моделей и их чекпоинтов
    """
    if not REGISTRY_PATH.exists():
        print(f"❌ Registry файл не найден: {REGISTRY_PATH}")
        return {}
    
    used_checkpoints = {}
    
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Пропускаем комментарии и пустые строки
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            folder = parts[0]
            checkpoint = parts[1]
            
            # Добавляем .zip если не указан
            if not checkpoint.endswith(".zip"):
                checkpoint = checkpoint + ".zip"
            
            used_checkpoints[folder] = checkpoint
    
    return used_checkpoints


def find_all_model_folders() -> List[Path]:
    """Находит все папки моделей в rl_system/models/."""
    if not MODELS_DIR.exists():
        print(f"❌ Директория моделей не найдена: {MODELS_DIR}")
        return []
    
    return [d for d in MODELS_DIR.iterdir() if d.is_dir()]


def get_best_checkpoint_from_json(model_path: Path) -> Optional[str]:
    """Извлекает имя лучшего чекпоинта из selected_best_by_metrics.json.
    
    Поддерживает два формата:
    1. Старый формат: {"model_path": "путь/к/чекпоинту.zip", ...}
    2. Новый формат: {"best_checkpoint": {"model_path": "путь/к/чекпоинту.zip", ...}, ...}
    
    Args:
        model_path: Путь к папке модели
        
    Returns:
        Имя файла чекпоинта (например, "rl_model_580000_steps.zip") или None
    """
    json_path = model_path / "selected_best_by_metrics.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Пытаемся найти model_path в разных форматах
        checkpoint_path_str = None
        
        # Формат 1: Новый формат с best_checkpoint
        if "best_checkpoint" in data and isinstance(data["best_checkpoint"], dict):
            checkpoint_path_str = data["best_checkpoint"].get("model_path")
            # Также проверяем поле checkpoint напрямую
            if not checkpoint_path_str:
                checkpoint_path_str = data["best_checkpoint"].get("checkpoint")
        
        # Формат 2: Старый формат - model_path в корне
        if not checkpoint_path_str:
            checkpoint_path_str = data.get("model_path")
        
        if not checkpoint_path_str:
            return None
        
        # Извлекаем только имя файла из полного пути
        checkpoint_path = Path(checkpoint_path_str)
        return checkpoint_path.name
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"  ⚠️  Ошибка чтения {json_path}: {e}")
        return None


def cleanup_checkpoints(dry_run: bool = False, auto_yes: bool = False) -> None:
    """Удаляет неиспользуемые чекпоинты.
    
    Args:
        dry_run: Если True, только показывает что будет удалено
        auto_yes: Если True, не спрашивает подтверждения
    """
    used_checkpoints = load_used_checkpoints()
    
    if not used_checkpoints:
        print("⚠️  Нет данных об используемых чекпоинтах в registry")
        return
    
    print(f"📋 Загружено {len(used_checkpoints)} моделей из registry\n")
    
    model_folders = find_all_model_folders()
    
    if not model_folders:
        print("⚠️  Не найдено папок с моделями")
        return
    
    total_size_to_free = 0
    total_files_to_delete = 0
    models_with_checkpoints = []
    
    # Первый проход: собираем информацию
    for model_path in model_folders:
        folder_name = model_path.name
        checkpoints_dir = model_path / "checkpoints"
        
        if not checkpoints_dir.exists():
            continue
        
        # Определяем какой чекпоинт нужно оставить:
        # 1. Если модель в registry - используем чекпоинт из registry
        # 2. Если модели нет в registry - используем чекпоинт из selected_best_by_metrics.json
        checkpoint_to_keep = used_checkpoints.get(folder_name)
        source = "registry"
        
        if not checkpoint_to_keep:
            checkpoint_to_keep = get_best_checkpoint_from_json(model_path)
            source = "selected_best_by_metrics.json"
        
        # Находим все чекпоинты
        all_checkpoints = list(checkpoints_dir.glob("*.zip"))
        
        if not all_checkpoints:
            continue
        
        # Определяем, какие чекпоинты нужно удалить
        to_delete = []
        kept = []
        
        for cp in all_checkpoints:
            if checkpoint_to_keep and cp.name == checkpoint_to_keep:
                kept.append(cp)
            else:
                to_delete.append(cp)
                total_size_to_free += cp.stat().st_size
                total_files_to_delete += 1
        
        # Проверяем best_model.zip и final_model.zip в корне модели
        extra_models_to_delete = []
        for model_file in ["best_model.zip", "final_model.zip"]:
            model_file_path = model_path / model_file
            if model_file_path.exists():
                extra_models_to_delete.append(model_file_path)
                total_size_to_free += model_file_path.stat().st_size
                total_files_to_delete += 1
        
        if to_delete or extra_models_to_delete:
            models_with_checkpoints.append({
                "folder": folder_name,
                "used": checkpoint_to_keep,
                "source": source,
                "to_delete": to_delete,
                "extra_to_delete": extra_models_to_delete,
                "kept": kept,
                "total_checkpoints": len(all_checkpoints)
            })
    
    if not models_with_checkpoints:
        print("✅ Нет чекпоинтов для удаления - все актуально!")
        return
    
    # Показываем статистику
    print(f"🔍 Найдено моделей с чекпоинтами для очистки: {len(models_with_checkpoints)}")
    print(f"📦 Всего файлов к удалению: {total_files_to_delete}")
    print(f"💾 Освободится места: {total_size_to_free / (1024**3):.2f} GB\n")
    
    # Показываем детали по каждой модели
    print("=" * 80)
    for model_info in models_with_checkpoints:
        print(f"\n📁 Модель: {model_info['folder']}")
        print(f"   Всего чекпоинтов: {model_info['total_checkpoints']}")
        
        if model_info['used']:
            print(f"   ✅ Сохраняется: {model_info['used']}")
            print(f"   📋 Источник: {model_info['source']}")
        else:
            print(f"   ⚠️  Лучший чекпоинт не найден (все чекпоинты будут удалены)")
        
        if model_info['to_delete']:
            size = sum(cp.stat().st_size for cp in model_info['to_delete'])
            print(f"   🗑️  Чекпоинты к удалению: {len(model_info['to_delete'])} файлов ({size / (1024**2):.1f} MB)")
        
        if model_info['extra_to_delete']:
            size = sum(f.stat().st_size for f in model_info['extra_to_delete'])
            extra_names = [f.name for f in model_info['extra_to_delete']]
            print(f"   🗑️  Дополнительно: {', '.join(extra_names)} ({size / (1024**2):.1f} MB)")
    
    print("\n" + "=" * 80)
    
    if dry_run:
        print("\n🔍 DRY RUN - файлы не будут удалены")
        return
    
    # Запрашиваем подтверждение
    if not auto_yes:
        print(f"\n⚠️  ВНИМАНИЕ! Будет удалено {total_files_to_delete} файлов.")
        print("   Останутся только лучшие выбранные чекпоинты")
        print("   best_model.zip и final_model.zip будут удалены")
        response = input("\n❓ Продолжить? (yes/no): ").strip().lower()
        
        if response not in ["yes", "y"]:
            print("❌ Отменено пользователем")
            return
    
    # Удаляем чекпоинты и дополнительные файлы
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    for model_info in models_with_checkpoints:
        print(f"\n📁 {model_info['folder']}")
        
        # Удаляем неиспользуемые чекпоинты
        for cp in model_info['to_delete']:
            try:
                size = cp.stat().st_size
                cp.unlink()
                deleted_count += 1
                deleted_size += size
                print(f"  🗑️  Удален чекпоинт: {cp.name}")
            except Exception as e:
                error_msg = f"Ошибка при удалении {cp}: {e}"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
        
        # Удаляем best_model.zip и final_model.zip
        for extra_file in model_info['extra_to_delete']:
            try:
                size = extra_file.stat().st_size
                extra_file.unlink()
                deleted_count += 1
                deleted_size += size
                print(f"  🗑️  Удален: {extra_file.name}")
            except Exception as e:
                error_msg = f"Ошибка при удалении {extra_file}: {e}"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
    
    # Итоговая статистика
    print("\n" + "=" * 80)
    print(f"\n✅ Успешно удалено: {deleted_count} файлов")
    print(f"💾 Освобождено места: {deleted_size / (1024**3):.2f} GB")
    
    if errors:
        print(f"\n⚠️  Ошибок: {len(errors)}")
        for error in errors:
            print(f"   - {error}")
    
    print("\n✅ Очистка завершена!")


def main():
    parser = argparse.ArgumentParser(
        description="Удаление неиспользуемых чекпоинтов RL моделей"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что будет удалено, но не удалять"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Не спрашивать подтверждения"
    )
    
    args = parser.parse_args()
    
    print("🧹 Скрипт очистки неиспользуемых чекпоинтов RL моделей")
    print("=" * 80)
    print(f"📂 Registry: {REGISTRY_PATH}")
    print(f"📂 Модели: {MODELS_DIR}")
    print("=" * 80 + "\n")
    
    if args.dry_run:
        print("🔍 Режим DRY RUN - файлы не будут удалены\n")
    
    cleanup_checkpoints(dry_run=args.dry_run, auto_yes=args.yes)


if __name__ == "__main__":
    main()
