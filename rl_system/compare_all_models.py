"""#!/usr/bin/env python3

Сравнение всех обученных RL моделей"""

====================================Compare All RL Models

Проверяет качество всех моделей в папке models/=====================

"""

Сравнивает производительность всех обученных RL моделей.

import json"""

from pathlib import Path

import sysimport os

import pandas as pdimport json

from pathlib import Path

sys.path.append('rl_system')from typing import List, Dict

from stable_baselines3 import PPO, A2C, SAC, TD3import subprocess



from data_loader import DataLoader

from feature_engineering import FeatureEngineerdef find_all_models() -> List[Path]:

from trading_env import MarketTradingEnv    """Находит все обученные модели."""

    models_dir = Path("rl_system/models")

    if not models_dir.exists():

def evaluate_model(model_path: Path):        print(f"❌ Models directory not found: {models_dir}")

    """Оценивает одну модель."""        return []

    try:    

        # Загружаем конфиг    models = []

        config_path = model_path / "config.json"    for model_dir in models_dir.iterdir():

        if not config_path.exists():        if model_dir.is_dir() and (model_dir / "config.json").exists():

            return None            models.append(model_dir)

                

        with open(config_path, 'r') as f:    return sorted(models, key=lambda x: x.name)

            config = json.load(f)

        

        # Определяем класс алгоритмаdef load_model_config(model_path: Path) -> Dict:

        algorithm = config['algorithm']    """Загружает конфигурацию модели."""

        algorithm_classes = {    config_path = model_path / "config.json"

            'PPO': PPO,    with open(config_path, 'r') as f:

            'A2C': A2C,        return json.load(f)

            'SAC': SAC,

            'TD3': TD3

        }def evaluate_model(model_path: Path, days: int = 730) -> Dict:

            """Оценивает модель на тестовых данных."""

        model_class = algorithm_classes.get(algorithm)    print(f"\n{'='*70}")

        if not model_class:    print(f"📊 Evaluating: {model_path.name}")

            print(f"⚠️  Unknown algorithm: {algorithm}")    print(f"{'='*70}\n")

            return None    

            cmd = [

        # Загружаем модель (пробуем best, потом final)        sys.executable,

        best_model_path = model_path / "best" / "best_model.zip"        "rl_system/evaluate_agent.py",

        final_model_path = model_path / "final_model.zip"        "--model-path", str(model_path),

                "--days", str(days),

        if best_model_path.exists():        "--deterministic"

            model = model_class.load(best_model_path)    ]

            model_type = "best"    

        elif final_model_path.exists():    result = subprocess.run(

            model = model_class.load(final_model_path)        cmd,

            model_type = "final"        capture_output=True,

        else:        text=True

            print(f"⚠️  No model files found")    )

            return None    

            # Parse output для извлечения метрик

        # Загружаем данные    # (упрощенная версия - можно улучшить)

        loader = DataLoader(cache_dir="data/cache")    output = result.stdout

        df = loader.load_data(    

            symbol=config['symbol'],    metrics = {

            timeframe=config['timeframe'],        'model_name': model_path.name,

            days=config['days']        'return': None,

        )        'trades': None,

                'win_rate': None,

        # Рассчитываем features        'max_dd': None

        engineer = FeatureEngineer()    }

        df = engineer.calculate_features(df)    

        df = engineer.normalize_features(df, method='zscore')    # Парсинг метрик из вывода

            for line in output.split('\n'):

        # Разделяем на train/test (80/20)        if 'Average Return:' in line:

        split_idx = int(len(df) * 0.8)            try:

        test_df = df.iloc[split_idx:].copy()                metrics['return'] = float(line.split(':')[1].strip().replace('%', ''))

                    except:

        # Создаем test окружение                pass

        test_env = MarketTradingEnv(        elif 'Average Trades:' in line:

            df=test_df,            try:

            initial_balance=10000.0,                metrics['trades'] = int(line.split(':')[1].strip())

            commission=config['commission'],            except:

            slippage=config['slippage'],                pass

            lookback_window=config['lookback'],        elif 'Average Win Rate:' in line:

            position_size=1.0,            try:

            enable_short=config.get('enable_short', True)                metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))

        )            except:

                        pass

        # Запускаем evaluation        elif 'Average Max DD:' in line:

        obs, _ = test_env.reset()            try:

        episode_reward = 0                metrics['max_dd'] = float(line.split(':')[1].strip().replace('%', ''))

        done = False            except:

                        pass

        while not done:    

            action, _ = model.predict(obs, deterministic=True)    return metrics

            obs, reward, terminated, truncated, info = test_env.step(action)

            episode_reward += reward

            done = terminated or truncateddef print_comparison_table(results: List[Dict]):

            """Выводит таблицу сравнения."""

        # Получаем финальные метрики    print("\n" + "="*100)

        final_info = test_env._get_info()    print("📊 MODEL COMPARISON RESULTS")

            print("="*100)

        return {    

            'model_name': model_path.name,    # Заголовок таблицы

            'algorithm': config['algorithm'],    print(f"\n{'Model':<40} {'Algorithm':<10} {'TF':<5} {'Return':<12} {'Trades':<8} {'Win Rate':<10} {'Max DD':<10}")

            'timeframe': config['timeframe'],    print("-" * 100)

            'days': config['days'],    

            'timesteps': config['total_timesteps'],    # Сортируем по доходности

            'model_type': model_type,    results_sorted = sorted(

            'return_pct': round(final_info['total_return_pct'], 2),        results,

            'final_balance': round(test_env.balance, 2),        key=lambda x: x.get('return', -9999),

            'total_trades': final_info['total_trades'],        reverse=True

            'win_rate': round(final_info['win_rate'] * 100, 1),    )

            'profit_factor': round(final_info['profit_factor'], 2),    

            'max_drawdown': round(final_info['max_drawdown_pct'], 2),    for i, result in enumerate(results_sorted, 1):

            'sharpe_ratio': round(final_info['sharpe_ratio'], 2),        # Извлекаем algorithm и timeframe из имени модели

            'episode_reward': round(episode_reward, 2)        model_name = result['model_name']

        }        parts = model_name.split('_')

                

    except Exception as e:        symbol = parts[0] if len(parts) > 0 else '?'

        print(f"❌ Error: {e}")        timeframe = parts[1] if len(parts) > 1 else '?'

        return None        algorithm = parts[2] if len(parts) > 2 else '?'

        

        ret = result['return']

def main():        ret_str = f"{ret:+.2f}%" if ret is not None else "N/A"

    """Сравнивает все модели в папке models/"""        

    models_dir = Path("rl_system/models")        trades = result['trades'] if result['trades'] is not None else "N/A"

            win_rate = f"{result['win_rate']:.1f}%" if result['win_rate'] is not None else "N/A"

    print(f"\n{'='*100}")        max_dd = f"{result['max_dd']:.1f}%" if result['max_dd'] is not None else "N/A"

    print(f"🔍 Comparing All RL Models")        

    print(f"{'='*100}\n")        # Emoji для топ-3

            emoji = ""

    results = []        if i == 1:

    model_folders = sorted([d for d in models_dir.iterdir() if d.is_dir()])            emoji = "🥇"

            elif i == 2:

    for i, model_path in enumerate(model_folders, 1):            emoji = "🥈"

        print(f"[{i}/{len(model_folders)}] Evaluating {model_path.name}...", end=" ")        elif i == 3:

                    emoji = "🥉"

        result = evaluate_model(model_path)        

        if result:        # Цвет для положительной доходности

            results.append(result)        if ret is not None and ret > 0:

            print(f"✅ {result['return_pct']:+.2f}% | PF={result['profit_factor']:.2f} | Trades={result['total_trades']}")            ret_str = f"✅ {ret_str}"

        else:        elif ret is not None:

            print("❌ Failed")            ret_str = f"❌ {ret_str}"

            

    if not results:        name_short = model_name[:37] + "..." if len(model_name) > 40 else model_name

        print("\n⚠️  No models evaluated successfully")        

        return        print(f"{emoji} {name_short:<38} {algorithm:<10} {timeframe:<5} {ret_str:<12} {trades:<8} {win_rate:<10} {max_dd:<10}")

        

    # Создаем DataFrame для красивого вывода    print("-" * 100)

    df = pd.DataFrame(results)    

        # Статистика

    # Сортируем по Return    profitable = sum(1 for r in results if r.get('return', -1) > 0)

    df = df.sort_values('return_pct', ascending=False)    total = len(results)

        

    print(f"\n{'='*100}")    print(f"\n📈 Summary:")

    print(f"📊 COMPARISON RESULTS (sorted by Return)")    print(f"   Total models: {total}")

    print(f"{'='*100}\n")    print(f"   Profitable: {profitable} ({profitable/total*100:.1f}%)")

        print(f"   Unprofitable: {total-profitable} ({(total-profitable)/total*100:.1f}%)")

    # Выводим таблицу    

    print(df.to_string(index=False))    if results_sorted and results_sorted[0].get('return'):

            best = results_sorted[0]

    # Топ-3 модели        print(f"\n🏆 Best Model:")

    print(f"\n{'='*100}")        print(f"   {best['model_name']}")

    print(f"🏆 TOP 3 MODELS")        print(f"   Return: {best['return']:+.2f}%")

    print(f"{'='*100}")        print(f"   Trades: {best['trades']}")

            print(f"   Win Rate: {best['win_rate']:.1f}%")

    for idx, (i, row) in enumerate(df.head(3).iterrows(), 1):    

        print(f"\n{idx}. {row['model_name']}")    print("\n" + "="*100 + "\n")

        print(f"   Algorithm: {row['algorithm']} ({row['timeframe']}, {row['timesteps']:,} steps)")

        print(f"   Return:    {row['return_pct']:+.2f}%")

        print(f"   PF:        {row['profit_factor']:.2f}")def main():

        print(f"   Trades:    {row['total_trades']} (WR: {row['win_rate']:.1f}%)")    """Главная функция."""

        print(f"   Sharpe:    {row['sharpe_ratio']:.2f}")    print("\n" + "="*70)

        print(f"   Max DD:    {row['max_drawdown']:.2f}%")    print("🤖 RL Trading Models - Performance Comparison")

        print("="*70 + "\n")

    # Худшие модели    

    print(f"\n{'='*100}")    # Находим все модели

    print(f"⚠️  WORST 3 MODELS (to delete)")    models = find_all_models()

    print(f"{'='*100}")    

        if not models:

    for idx, (i, row) in enumerate(df.tail(3).iterrows(), 1):        print("❌ No trained models found in rl_system/models/")

        print(f"\n{idx}. {row['model_name']}")        print("💡 Train models first using: python rl_system/train_best_models.py")

        print(f"   Algorithm: {row['algorithm']} ({row['timeframe']}, {row['timesteps']:,} steps)")        return

        print(f"   Return:    {row['return_pct']:+.2f}%")    

        print(f"   PF:        {row['profit_factor']:.2f}")    print(f"Found {len(models)} trained models:\n")

        for i, model in enumerate(models, 1):

    # Статистика по алгоритмам        config = load_model_config(model)

    print(f"\n{'='*100}")        algo = config.get('algorithm', '?')

    print(f"📈 STATISTICS BY ALGORITHM")        tf = config.get('timeframe', '?')

    print(f"{'='*100}\n")        print(f"{i}. {model.name} ({algo}, {tf})")

        

    algo_stats = df.groupby('algorithm').agg({    # Спрашиваем пользователя

        'return_pct': ['mean', 'max', 'min', 'count'],    print("\n" + "-"*70)

        'profit_factor': 'mean',    choice = input("\nEvaluate all models? (y/n): ").strip().lower()

        'total_trades': 'mean'    

    }).round(2)    if choice != 'y':

            print("\n👋 Cancelled by user")

    print(algo_stats.to_string())        return

        

    print(f"\n{'='*100}\n")    # Оцениваем все модели

    results = []

    for model in models:

if __name__ == '__main__':        try:

    main()            metrics = evaluate_model(model)

            results.append(metrics)
        except Exception as e:
            print(f"❌ Error evaluating {model.name}: {e}")
            continue
    
    # Выводим таблицу сравнения
    if results:
        print_comparison_table(results)
        
        # Сохраняем результаты
        results_file = Path("rl_system/model_comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"💾 Results saved to: {results_file}")
    else:
        print("\n❌ No results to display")


if __name__ == "__main__":
    main()
