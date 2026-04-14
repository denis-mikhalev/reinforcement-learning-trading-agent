import csv
import argparse
from pathlib import Path


def analyze_trades_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    closes = [r for r in rows if r.get("event") == "CLOSE"]

    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0

    for r in closes:
        pnl = float(r.get("pnl", 0.0))
        total_pnl += pnl
        if pnl > 0:
            gross_profit += pnl
            wins += 1
        elif pnl < 0:
            gross_loss += -pnl
            losses += 1

    total_trades = len(closes)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    first_equity = float(closes[0]["equity_after"]) if closes else 0.0
    last_equity = float(closes[-1]["equity_after"]) if closes else 0.0
    total_return_pct = ((last_equity / first_equity - 1.0) * 100.0) if first_equity > 0 else 0.0

    print(f"File: {path}")
    print(f"Total trades (CLOSE events): {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}, Win rate: {win_rate:.2f}%")
    print(f"Gross profit: {gross_profit:.2f}, Gross loss: {gross_loss:.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"First equity: {first_equity:.2f}, Last equity: {last_equity:.2f}")
    print(f"Total return (from equity series): {total_return_pct:.2f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trades CSV exported from CryptoTradingEnv.")
    parser.add_argument("csv_path", type=str, help="Path to trades CSV file")
    args = parser.parse_args()

    analyze_trades_csv(Path(args.csv_path))
