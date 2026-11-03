# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Training evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(labels: np.ndarray, predictions: np.ndarray, quantile: float) -> dict[str, float]:
    if len(labels) == 0:
        return {"pinballLoss": 0.0, "coverage": 0.0, "violationRate": 0.0, "mae": 0.0, "rmse": 0.0}
    errors = labels - predictions
    pinball = np.maximum(quantile * errors, (quantile - 1.0) * errors)
    coverage = float(np.mean(labels <= predictions))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "pinballLoss": float(np.mean(pinball)),
        "coverage": coverage,
        "violationRate": 1.0 - coverage,
        "mae": mae,
        "rmse": rmse,
    }


def ranking_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Rank-fidelity of predictions vs actual latency on a holdout slice.

    These ordering metrics answer "does the model rank latencies in the right
    order", which is what routing relies on, independent of absolute calibration.
    Returns ``valid=False`` (with None scores) when there is no determinable
    ordering — fewer than two samples, or all labels/predictions tied — so the
    report never presents a misleading 0. This is a weak proxy: it ranks across
    different requests, so it mixes prompt-length and load effects rather than
    isolating same-request endpoint ordering; read the input-length strata to
    approximate the load-only signal.
    """
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    n = int(labels.shape[0])
    invalid = {"valid": False, "n": n, "spearman": None, "pairwiseAccuracy": None, "pairs": 0}
    if n < 2:
        return invalid
    spearman = _spearman(labels, predictions)
    pairwise, pairs = _pairwise_accuracy(labels, predictions, max_pairs, seed)
    if spearman is None or pairwise is None:
        return {**invalid, "spearman": spearman, "pairwiseAccuracy": pairwise, "pairs": pairs}
    return {"valid": True, "n": n, "spearman": spearman, "pairwiseAccuracy": pairwise, "pairs": pairs}


def stratified_ranking_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    strata: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compute `ranking_metrics` over the full slice plus each named stratum mask."""
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    result = {"overall": ranking_metrics(labels, predictions)}
    for name, mask in strata.items():
        mask = np.asarray(mask, dtype=bool)
        result[name] = ranking_metrics(labels[mask], predictions[mask])
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties broken by their average position (Spearman tie handling)."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=float)
    index = 0
    total = values.shape[0]
    while index < total:
        end = index
        while end + 1 < total and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0
        index = end + 1
    return ranks


def _spearman(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    return _pearson(_average_ranks(labels), _average_ranks(predictions))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left_std = left.std()
    right_std = right.std()
    if left_std == 0.0 or right_std == 0.0:
        return None
    centered = (left - left.mean()) * (right - right.mean())
    return float(centered.mean() / (left_std * right_std))


def _pairwise_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    max_pairs: int,
    seed: int,
) -> tuple[float | None, int]:
    """Fraction of comparable pairs the model orders correctly (concordance).

    Pairs with equal labels carry no ground-truth order and are skipped. Above
    `max_pairs` total pairs we sample with a fixed seed to stay out of O(n^2).
    """
    n = labels.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        left, right = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, n, size=max_pairs)
        right = rng.integers(0, n, size=max_pairs)
    comparable = (left != right) & (labels[left] != labels[right])
    left = left[comparable]
    right = right[comparable]
    counted = int(left.shape[0])
    if counted == 0:
        return None, 0
    concordant = int(np.count_nonzero((predictions[left] > predictions[right]) == (labels[left] > labels[right])))
    return concordant / counted, counted
