# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Artifact publication for trained prediction models."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.models import TrainingSample
from schema import SLOTS, columns_for_slot, monotone_constraints_for
from training import lightgbm_backend, xgboost_backend
from training.features import labels_for_samples, matrix_for_samples
from training.metrics import regression_metrics, stratified_ranking_metrics
from training.split import capped_bucket_sample, temporal_holdout


@dataclass(frozen=True)
class TrainOptions:
    artifact_root: Path
    target_model: str
    model_version: str
    model_backend: str
    quantile: float
    min_samples_per_slot: int
    min_holdout_samples_per_slot: int
    holdout_ratio: float
    bucket_cap: int
    hyperparams: dict[str, Any] = field(default_factory=dict)
    enable_monotone_constraints: bool = True


def write_artifacts(
    samples: list[TrainingSample],
    drop_counts: Counter[str],
    options: TrainOptions,
) -> Path:
    if options.model_backend not in {"xgboost", "lightgbm"}:
        raise ValueError("model_backend must be xgboost or lightgbm")
    if not options.target_model:
        raise ValueError("target_model is required")
    if not options.model_version:
        raise ValueError("model_version is required")

    bundle_dir = options.artifact_root / options.target_model / options.model_version
    staging_dir = options.artifact_root / options.target_model / f".staging-{options.model_version}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "slots").mkdir(parents=True, exist_ok=True)

    slot_groups = group_by_slot([item for item in samples if item.target_model == options.target_model])
    promoted_slots: list[str] = []
    slot_files: dict[str, str] = {}
    slot_feature_columns: dict[str, list[str]] = {}
    slot_monotone: dict[str, list[int]] = {}
    slot_statuses: dict[str, dict] = {}
    report_slots: dict[str, dict] = {}

    for slot in SLOTS:
        columns = columns_for_slot(slot)
        slot_samples = slot_groups.get(slot, [])
        train, holdout = temporal_holdout(slot_samples, options.holdout_ratio)
        train = capped_bucket_sample(train, options.bucket_cap)
        status = {
            "status": "below_threshold",
            "samples": len(slot_samples),
            "trainSamples": len(train),
            "holdoutSamples": len(holdout),
            "failingThreshold": "",
        }
        if len(slot_samples) < options.min_samples_per_slot:
            status["failingThreshold"] = "min_samples_per_slot"
            slot_statuses[slot] = status
            continue
        if len(holdout) < options.min_holdout_samples_per_slot:
            status["failingThreshold"] = "min_holdout_samples_per_slot"
            slot_statuses[slot] = status
            continue

        # LightGBM rejects monotone_constraints under the quantile objective, so
        # constraints apply to the XGBoost backend only; LightGBM slots record [].
        monotone = (
            monotone_constraints_for(columns)
            if options.enable_monotone_constraints and options.model_backend == "xgboost"
            else None
        )
        model = fit_model(
            options.model_backend,
            matrix_for_samples(train, columns),
            labels_for_samples(train),
            options.quantile,
            options.hyperparams,
            monotone,
        )
        holdout_labels = labels_for_samples(holdout)
        predictions = predict_model(options.model_backend, model, matrix_for_samples(holdout, columns))
        metrics = regression_metrics(holdout_labels, predictions, options.quantile)
        ranking = stratified_ranking_metrics(holdout_labels, predictions, holdout_strata(holdout))
        extension = "xgboost.json" if options.model_backend == "xgboost" else "lightgbm.txt"
        relative_model_path = f"slots/{slot}.{extension}"
        save_model(options.model_backend, model, staging_dir / relative_model_path)
        promoted_slots.append(slot)
        slot_files[slot] = relative_model_path
        slot_feature_columns[slot] = list(columns)
        slot_monotone[slot] = list(monotone) if monotone is not None else []
        status.update({"status": "promoted", "failingThreshold": "", **metrics})
        slot_statuses[slot] = status
        report_slots[slot] = {
            **metrics,
            "ranking": ranking,
            "trainSamples": len(train),
            "holdoutSamples": len(holdout),
            "samples": len(slot_samples),
        }

    manifest = {
        "bundleVersion": 2,
        "targetModel": options.target_model,
        "modelVersion": options.model_version,
        "backend": options.model_backend,
        "quantile": options.quantile,
        "promotedSlots": promoted_slots,
        "slotFiles": slot_files,
        "slotFeatureColumns": slot_feature_columns,
    }
    metadata = {
        "targetModel": options.target_model,
        "modelVersion": options.model_version,
        "backend": options.model_backend,
        "quantile": options.quantile,
        "slotFeatureColumns": slot_feature_columns,
        "slotMonotoneConstraints": slot_monotone,
        "monotoneConstraintsEnabled": options.enable_monotone_constraints,
        "thresholds": {
            "minSamplesPerSlot": options.min_samples_per_slot,
            "minHoldoutSamplesPerSlot": options.min_holdout_samples_per_slot,
        },
        "holdout": {"ratio": options.holdout_ratio, "bucketCap": options.bucket_cap},
        "hyperparams": dict(options.hyperparams),
        "slots": slot_statuses,
    }
    report = {
        "targetModel": options.target_model,
        "modelVersion": options.model_version,
        "backend": options.model_backend,
        "quantile": options.quantile,
        "slots": report_slots,
        "dropReasons": dict(drop_counts),
    }
    write_json(staging_dir / "manifest.json", manifest)
    write_json(staging_dir / "metadata.json", metadata)
    write_json(staging_dir / "report.json", report)

    publish_bundle(staging_dir, bundle_dir, options)
    return bundle_dir


def publish_bundle(staging_dir: Path, bundle_dir: Path, options: TrainOptions) -> None:
    """Atomically swap the staging dir into place via rename-to-backup.

    Spec invariant: the sidecar must never observe a partially written bundle.
    POSIX rename only atomically replaces an empty target, so we cannot rename
    `staging_dir` directly onto a populated `bundle_dir`. Instead we move the
    existing bundle to a backup name (single rename), then rename staging into
    place (single rename), then remove the backup. The window during which
    `bundle_dir` does not exist is bounded by two rename syscalls. A sidecar
    starting in that window reads no `manifest.json` and falls back to the
    not-prediction-ready path; it never sees a half-written bundle.
    """
    backup_dir: Path | None = None
    if bundle_dir.exists():
        backup_dir = (
            options.artifact_root
            / options.target_model
            / f".backup-{options.model_version}-{int(time.time_ns())}"
        )
        bundle_dir.rename(backup_dir)
    try:
        staging_dir.rename(bundle_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not bundle_dir.exists():
            # Roll back so the previous bundle keeps serving rather than
            # leaving the live path empty.
            backup_dir.rename(bundle_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def group_by_slot(samples: list[TrainingSample]) -> dict[str, list[TrainingSample]]:
    grouped: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.slot].append(sample)
    return grouped


def holdout_strata(holdout: list[TrainingSample]) -> dict[str, np.ndarray]:
    """Boolean stratum masks (aligned to holdout order) for ranking diagnostics.

    `queued`/`noQueue` show whether the unified model learned the queue effect.
    The input-length strata approximate "rank by load with the prompt held
    fixed" — the per-decision signal routing actually depends on — and are only
    emitted once there are enough samples to split meaningfully.
    """
    waiting = np.array([int(item.prediction_input.get("numRequestWaiting") or 0) for item in holdout])
    strata: dict[str, np.ndarray] = {
        "queued": waiting > 0,
        "noQueue": waiting == 0,
    }
    if len(holdout) >= 4:
        input_len = np.array([float(item.prediction_input.get("inputTokenLength") or 0.0) for item in holdout])
        inflight = np.array([float(item.prediction_input.get("inflightTokens") or 0.0) for item in holdout])
        len_median = float(np.median(input_len))
        load_median = float(np.median(inflight))
        strata["inputLenLow"] = input_len <= len_median
        strata["inputLenHigh"] = input_len > len_median
        strata["loadLow"] = inflight <= load_median
        strata["loadHigh"] = inflight > load_median
    return strata


def fit_model(
    backend: str,
    features: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    overrides: dict[str, Any] | None = None,
    monotone_constraints: tuple[int, ...] | None = None,
) -> Any:
    if backend == "xgboost":
        return xgboost_backend.fit_quantile_model(features, labels, quantile, overrides, monotone_constraints)
    return lightgbm_backend.fit_quantile_model(features, labels, quantile, overrides, monotone_constraints)


def predict_model(backend: str, model: Any, features: np.ndarray) -> np.ndarray:
    if backend == "xgboost":
        return xgboost_backend.predict(model, features)
    return lightgbm_backend.predict(model, features)


def save_model(backend: str, model: Any, path: Path) -> None:
    if backend == "xgboost":
        xgboost_backend.save_model(model, path)
        return
    lightgbm_backend.save_model(model, path)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
