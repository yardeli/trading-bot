"""
Auto-Trader — Regime-Adaptive Crypto Multi-Strategy
====================================================
Combines momentum and mean reversion signals with ML ensemble
prediction and Kelly criterion sizing.

Usage:
    python main.py                  # Full pipeline: fetch → features → backtest
    python main.py --cpcv           # Run CPCV validation instead
    python main.py --quick          # Quick backtest with fewer assets
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import INITIAL_CAPITAL, OUTPUT_DIR
from data.fetcher import fetch_ohlcv, build_universe
from features.engineering import (
    build_features_universe, build_cross_sectional_features,
    prepare_ml_dataset,
)
from backtest.walk_forward import WalkForwardBacktester, CPCVBacktester
from backtest.metrics import print_results


def run_pipeline(args):
    """Full pipeline: fetch data → build features → backtest."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # ── Step 1: Fetch Data ──────────────────────────────────────────
    print("\n[1/4] Fetching OHLCV data...")
    assets = None
    if args.quick:
        from config import ASSETS
        assets = ASSETS[:5]  # Top 5 only for quick mode
        print(f"  Quick mode: using {len(assets)} assets")

    data = fetch_ohlcv(symbols=assets)
    if not data:
        print("ERROR: No data fetched. Check internet connection.")
        return

    universe = build_universe(data)

    # ── Step 2: Feature Engineering ─────────────────────────────────
    print("\n[2/4] Building features...")
    features = build_features_universe(data)
    features = build_cross_sectional_features(features, universe)
    print(f"  Features per asset: {len(next(iter(features.values())).columns)}")

    # ── Step 3: Prepare ML Dataset ──────────────────────────────────
    print("\n[3/4] Preparing ML dataset...")
    dataset = prepare_ml_dataset(features, data, forward_days=7)
    print(f"  Dataset: {len(dataset)} samples, {dataset['symbol'].nunique()} assets")
    print(f"  Target distribution: {dataset['target_up'].mean():.1%} positive")
    print(f"  Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}")

    # ── Step 4: Backtest ────────────────────────────────────────────
    if args.cpcv:
        print("\n[4/4] Running CPCV backtest...")
        cpcv = CPCVBacktester()
        results = cpcv.run(dataset, features, data)
        print(f"\n  CPCV Results:")
        print(f"  Mean Accuracy: {results.get('mean_accuracy', 0):.1f}%")
        print(f"  Std Accuracy:  {results.get('std_accuracy', 0):.1f}%")
        print(f"  Min Accuracy:  {results.get('min_accuracy', 0):.1f}%")
        print(f"  Max Accuracy:  {results.get('max_accuracy', 0):.1f}%")
    else:
        print("\n[4/4] Running walk-forward backtest...")
        backtester = WalkForwardBacktester()
        results = backtester.run(dataset, features, data)
        print_results(results)

    # ── Save Results ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 1)
    results["timestamp"] = datetime.now().isoformat()
    results["mode"] = "cpcv" if args.cpcv else "walk_forward"

    # Remove non-serializable items
    clean_results = {k: v for k, v in results.items()
                     if not isinstance(v, list) or (v and isinstance(v[0], dict))}

    output_path = os.path.join(OUTPUT_DIR, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-Trader Backtest Pipeline")
    parser.add_argument("--cpcv", action="store_true", help="Run CPCV validation")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer assets)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AUTO-TRADER: Regime-Adaptive Crypto Multi-Strategy")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 60)

    run_pipeline(args)


if __name__ == "__main__":
    main()
