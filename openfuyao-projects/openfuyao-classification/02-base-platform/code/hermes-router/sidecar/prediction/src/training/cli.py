# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CLI for offline prediction training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.artifacts import TrainOptions, write_artifacts
from training.jsonl_loader import load_completed_jsonl
from training.samples import generate_samples

# Spec-baseline defaults from the design's "Dataset Split, Eligibility, And Slot Sampling" section.
# Each can be lowered for fixture-sized tests by passing CLI overrides.
DEFAULT_MIN_SAMPLES_PER_SLOT = 1000
DEFAULT_MIN_HOLDOUT_SAMPLES_PER_SLOT = 200
DEFAULT_HOLDOUT_RATIO = 0.2
DEFAULT_BUCKET_CAP = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hermes prediction artifacts")
    parser.add_argument("--input", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-backend", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--min-samples-per-slot", type=int, default=DEFAULT_MIN_SAMPLES_PER_SLOT)
    parser.add_argument(
        "--min-holdout-samples-per-slot",
        type=int,
        default=DEFAULT_MIN_HOLDOUT_SAMPLES_PER_SLOT,
    )
    parser.add_argument("--holdout-ratio", type=float, default=DEFAULT_HOLDOUT_RATIO)
    parser.add_argument("--bucket-cap", type=int, default=DEFAULT_BUCKET_CAP)
    parser.add_argument(
        "--enable-monotone-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply schema-defined monotone constraints to strong-semantic features (default on).",
    )
    parser.add_argument(
        "--hyperparams",
        default="{}",
        help=(
            "Backend-specific hyperparameter overrides as a JSON object. "
            "Merges over the spec baseline defined in the backend module."
        ),
    )
    return parser.parse_args()


def parse_hyperparams(raw: str) -> dict[str, Any]:
    decoded = json.loads(raw) if raw else {}
    if not isinstance(decoded, dict):
        raise ValueError("--hyperparams must decode to a JSON object")
    return decoded


def main() -> None:
    args = parse_args()
    hyperparams = parse_hyperparams(args.hyperparams)
    records, load_drops = load_completed_jsonl(Path(args.input))
    samples, sample_drops = generate_samples(records)
    drop_counts: Counter[str] = Counter()
    drop_counts.update(load_drops)
    drop_counts.update(sample_drops)
    bundle = write_artifacts(
        samples,
        drop_counts,
        TrainOptions(
            artifact_root=Path(args.artifact_root),
            target_model=args.target_model,
            model_version=args.model_version,
            model_backend=args.model_backend,
            quantile=args.quantile,
            min_samples_per_slot=args.min_samples_per_slot,
            min_holdout_samples_per_slot=args.min_holdout_samples_per_slot,
            holdout_ratio=args.holdout_ratio,
            bucket_cap=args.bucket_cap,
            hyperparams=hyperparams,
            enable_monotone_constraints=args.enable_monotone_constraints,
        ),
    )
    print(bundle)


if __name__ == "__main__":
    main()
