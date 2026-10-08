#!/usr/bin/env python3
"""CLI entry point for the complete reproducible experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import PipelineConfig
from src.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud detection and drift monitoring experiments")
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"), help="Path to Kaggle creditcard.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Repository/output root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp-cost", type=float, default=5.0, help="Expected cost per false positive (USD-equivalent)")
    parser.add_argument("--fn-cost", type=float, default=None, help="Override FN expected cost; default derives it from training fraud amounts")
    parser.add_argument("--skip-retraining-demo", action="store_true", help="Skip the optional synthetic concept-drift retraining prototype")
    parser.add_argument("--skip-model-saving", action="store_true", help="Do not write joblib model artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        random_seed=args.seed,
        false_positive_cost=args.fp_cost,
        false_negative_cost_override=args.fn_cost,
        run_retraining_demo=not args.skip_retraining_demo,
        save_models=not args.skip_model_saving,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
