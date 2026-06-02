#!/usr/bin/env python3
"""Build low-price classifier sequence dataset (28 features) under data/model_low_price."""

from run_sequence import _print_report
from sequence.pipeline import run_pipeline


def main() -> None:
    report = run_pipeline(feature_profile="low_price_classifier")
    _print_report(report)


if __name__ == "__main__":
    main()
