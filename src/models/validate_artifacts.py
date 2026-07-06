#!/usr/bin/env python3
"""Sanity-check delay model artifacts after training.

Usage:
    python -m src.models.validate_artifacts
    python -m src.models.validate_artifacts --path artifacts/global
"""

import argparse
import json
import sys
from pathlib import Path

QUANTILE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
CATEGORICAL_COLS = ["bpuic", "line_text", "hour", "day_of_week", "month", "day_type"]

EXPECTED_FILES = [
    "delay_model_metadata.json",
    "hist_aggs.json",
    "climatological_weather.json",
    "stop_station_mapping.json",
    "precomputed_delays.json",
] + [f"xgb_model_q{int(round(q * 100)):03d}.json" for q in QUANTILE_LEVELS]


def _check(results: list, name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)
    return condition


def _check_dict(results: list, data: object, label: str) -> bool:
    ok = _check(results, f"{label} is a dict", isinstance(data, dict))
    if ok:
        _check(results, f"{label} non-empty", len(data) > 0, f"{len(data):,} entries")
    return ok


def _check_metadata(results: list, path: Path) -> None:
    print("\n=== delay_model_metadata.json ===")
    meta = json.loads((path / "delay_model_metadata.json").read_text())

    for field in ("train_start_date", "train_end_date", "val_start_date", "val_end_date"):
        _check(results, field, field in meta, meta.get(field, "MISSING"))

    q_levels = meta.get("quantile_levels", [])
    _check(results, "quantile_levels count", len(q_levels) == 7, f"{len(q_levels)} found: {q_levels}")

    pinball = meta.get("val_pinball_loss", {})
    _check(results, "val_pinball_loss present", len(pinball) > 0, f"{len(pinball)} entries")
    for k, v in pinball.items():
        _check(results, f"  pinball[{k}] > 0", isinstance(v, (int, float)) and v > 0, f"{v:.4f}")

    cat_cols = meta.get("categorical_cols", [])
    _check(results, "categorical_cols", cat_cols == CATEGORICAL_COLS, str(cat_cols))


def _check_precomputed_delays(results: list, path: Path) -> None:
    print("\n=== precomputed_delays.json ===")
    delays = json.loads((path / "precomputed_delays.json").read_text())

    if not _check_dict(results, delays, "precomputed_delays"):
        return

    sample_key = next(iter(delays))
    sample_val = delays[sample_key]
    _check(results, "key has 6 parts (bpuic__line__hour__dow__month__day_type)",
           len(sample_key.split("__")) == 6, f"sample: {sample_key!r}")
    _check(results, "value is list of 7 floats",
           isinstance(sample_val, list) and len(sample_val) == 7, str(sample_val))

    sample_lists = list(delays.values())[:1000]
    out_of_range = sum(1 for lst in sample_lists for v in lst if not (-300 <= v <= 3600))
    _check(results, "values in sane range [-300s, 3600s]", out_of_range == 0,
           f"{out_of_range} violations in first 1 000 combos" if out_of_range else "OK")

    non_monotone = sum(
        1 for lst in sample_lists
        if any(lst[i] > lst[i + 1] + 1e-6 for i in range(len(lst) - 1))
    )
    _check(results, "quantiles non-decreasing per combo", non_monotone == 0,
           f"{non_monotone} violations in first 1 000 combos" if non_monotone else "OK")


def _check_supporting_jsons(results: list, path: Path) -> None:
    for fname, label in [
        ("hist_aggs.json", "hist_aggs"),
        ("stop_station_mapping.json", "stop_station_mapping"),
        ("climatological_weather.json", "climatological_weather"),
    ]:
        print(f"\n=== {fname} ===")
        data = json.loads((path / fname).read_text())
        _check_dict(results, data, label)


def _check_xgb_models(results: list, path: Path) -> None:
    print("\n=== XGBoost model files ===")
    try:
        import xgboost as xgb  # type: ignore[import]
    except ImportError:
        print("  [SKIP] xgboost not installed — skipping model load checks")
        return

    for q in QUANTILE_LEVELS:
        q_str = f"q{int(round(q * 100)):03d}"
        try:
            booster = xgb.Booster()
            booster.load_model(str(path / f"xgb_model_{q_str}.json"))
            _check(results, f"xgb_model_{q_str}.json loads OK", True,
                   f"{booster.num_boosted_rounds()} trees")
        except Exception as exc:
            _check(results, f"xgb_model_{q_str}.json loads OK", False, str(exc))


def validate(path: Path) -> bool:
    print(f"\nArtifact directory: {path}\n")
    results: list[bool] = []

    print("=== File existence ===")
    for fname in EXPECTED_FILES:
        fpath = path / fname
        exists = fpath.exists()
        size = f"{fpath.stat().st_size / 1024:.0f} KB" if exists else ""
        _check(results, fname, exists, size)

    if not all(results):
        print("\nSome files are missing — skipping content checks.")
        return False

    _check_metadata(results, path)
    _check_precomputed_delays(results, path)
    _check_supporting_jsons(results, path)
    _check_xgb_models(results, path)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 45}")
    if n_pass == n_total:
        print(f"All {n_total} checks PASSED. Artifacts look good.")
    else:
        print(f"{n_pass}/{n_total} checks passed — {n_total - n_pass} FAILED.")
    return n_pass == n_total


def main():
    parser = argparse.ArgumentParser(description="Validate delay model training artifacts.")
    parser.add_argument(
        "--path", default="artifacts/global",
        help="Path to the global artifact directory (default: artifacts/global)",
    )
    args = parser.parse_args()
    sys.exit(0 if validate(Path(args.path)) else 1)


if __name__ == "__main__":
    main()
