# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Feature engineering for training and runtime prediction.

Two builders share one feature contract: a scalar path (`feature_row_for_input`)
that doubles as the parity oracle, and a vectorized batch path
(`feature_matrix_for_inputs`) that the training and runtime hot paths use. Both
read raw fields through `_numeric` and bucket through the same thresholds, so
neither can drift into train/serve feature skew.
"""

from __future__ import annotations

import math

import numpy as np

from domain.models import TrainingSample
from schema import ALL_FEATURE_COLUMNS, DIRECT_FEATURE_COLUMNS

# Bucket thresholds shared by the scalar and vectorized builders. Equal-to-threshold
# values move up a bucket (the [lo, hi) convention encoded by `bucket()` and by
# `np.searchsorted(..., side="right")`).
PREFIX_CACHE_BUCKET_THRESHOLDS = [0.25, 0.5, 0.75]  # 0:[0,.25) 1:[.25,.5) 2:[.5,.75) 3:[.75,1]
QUEUE_BUCKET_THRESHOLDS = [1.0, 3.0, 6.0, 11.0]  # 0:idle 1:1-2 2:3-5 3:6-10 4:>=11
# kvCacheUsage in [0,1] is split into KV_CACHE_BUCKET_COUNT bins; out-of-range usage
# clamps to [0, KV_CACHE_BUCKET_MAX] rather than overflowing the bucket index.
KV_CACHE_BUCKET_COUNT = 20
KV_CACHE_BUCKET_MAX = KV_CACHE_BUCKET_COUNT - 1


def _numeric(value: object) -> float:
    """Coerce a raw feature value to float under the shared nil/falsey contract.

    Missing, present-None, and falsey (`0`, `""`) values all resolve to `0.0`.
    Non-finite values (NaN / inf / -inf, which proto doubles can carry) are
    rejected with `ValueError` so the gRPC layer reports INVALID_ARGUMENT instead
    of silently bucketing them or feeding inf into the model.

    This is the single repo-level contract for invalid numeric input; both the
    scalar projection and the vectorized extraction route through it, so neither
    a present-None nor a non-finite value can diverge between the two paths.
    """
    number = float(value or 0.0)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric feature value: {value!r}")
    return number


def engineered_features(prediction_input: dict) -> dict:
    """Augment a raw prediction input with the derived feature columns.

    Split out from `feature_row_for_input` so the runtime can compute the
    engineered row once per backend and project it onto multiple slot column
    sets (TTFT/TPOT) without re-running the bucket math for each slot.
    """
    engineered = dict(prediction_input)
    prefix = _numeric(engineered.get("prefixCacheScore"))
    input_tokens = _numeric(engineered.get("inputTokenLength"))
    waiting = int(_numeric(engineered.get("numRequestWaiting")))
    kv_usage = _numeric(engineered.get("kvCacheUsage"))
    engineered["effectiveInputTokens"] = input_tokens * max(0.0, 1.0 - prefix)
    engineered["prefixCacheBucket"] = bucket(prefix, PREFIX_CACHE_BUCKET_THRESHOLDS)
    engineered["queueBucket"] = bucket(float(waiting), QUEUE_BUCKET_THRESHOLDS)
    engineered["kvCacheUsageBucket"] = float(min(KV_CACHE_BUCKET_MAX, max(0, int(kv_usage * KV_CACHE_BUCKET_COUNT))))
    # isQueued: binary idle/queued split signal (see DERIVED_FEATURE_COLUMNS note).
    engineered["isQueued"] = 1.0 if waiting > 0 else 0.0
    return engineered


def project_columns(engineered: dict, columns: list[str] | None = None) -> list[float]:
    """Select `columns` from an engineered row under the shared numeric contract."""
    return [_numeric(engineered.get(column)) for column in (columns or ALL_FEATURE_COLUMNS)]


def feature_row_for_input(prediction_input: dict, columns: list[str] | None = None) -> list[float]:
    return project_columns(engineered_features(prediction_input), columns)


def engineered_feature_columns(prediction_inputs: list[dict]) -> dict[str, np.ndarray]:
    """Vectorized analog of `engineered_features` over many inputs.

    Returns a `{column_name: 1-D float array}` map covering every direct and
    derived feature, engineered exactly once per input. The runtime projects this
    one engineered batch onto multiple slot column sets (TTFT/TPOT), so the bucket
    math is not recomputed per slot — the vectorized equivalent of the scalar
    "engineer once, project twice" optimization.
    """
    columns: dict[str, np.ndarray] = {key: _numeric_column(prediction_inputs, key) for key in DIRECT_FEATURE_COLUMNS}
    prefix = columns["prefixCacheScore"]
    input_tokens = columns["inputTokenLength"]
    kv_usage = columns["kvCacheUsage"]
    # int() truncates toward zero; np astype(int64) matches that for derived queue
    # features while the direct numRequestWaiting column keeps its raw float value.
    waiting = columns["numRequestWaiting"].astype(np.int64)
    columns["effectiveInputTokens"] = input_tokens * np.maximum(0.0, 1.0 - prefix)
    columns["prefixCacheBucket"] = _bucketize(prefix, PREFIX_CACHE_BUCKET_THRESHOLDS)
    columns["queueBucket"] = _bucketize(waiting.astype(float), QUEUE_BUCKET_THRESHOLDS)
    columns["kvCacheUsageBucket"] = _kv_cache_usage_bucket(kv_usage)
    columns["isQueued"] = (waiting > 0).astype(float)
    return columns


def project_columns_matrix(engineered_columns: dict[str, np.ndarray], columns: list[str]) -> np.ndarray:
    """Stack the requested engineered columns into an (n, len(columns)) matrix.

    Columns absent from `engineered_columns` stay at the `0.0` default, matching
    the scalar projection's missing-key behavior.
    """
    row_count = _column_length(engineered_columns)
    matrix = np.zeros((row_count, len(columns)), dtype=float)
    for index, column in enumerate(columns):
        values = engineered_columns.get(column)
        if values is not None:
            matrix[:, index] = values
    return matrix


def feature_matrix_for_inputs(prediction_inputs: list[dict], columns: list[str] | None = None) -> np.ndarray:
    """Build a feature matrix for many inputs at once, matching scalar semantics.

    Preferred batch path for training matrices and runtime slot matrices. Returns
    a `(len(prediction_inputs), len(columns))` array; an empty input list yields a
    `(0, width)` matrix so callers can predict/reshape without a 1-D collapse.
    """
    selected = list(columns) if columns is not None else ALL_FEATURE_COLUMNS
    if not prediction_inputs:
        return np.zeros((0, len(selected)), dtype=float)
    return project_columns_matrix(engineered_feature_columns(prediction_inputs), selected)


def matrix_for_samples(samples: list[TrainingSample], columns: list[str] | None = None) -> np.ndarray:
    return feature_matrix_for_inputs([sample.prediction_input for sample in samples], columns)


def labels_for_samples(samples: list[TrainingSample]) -> np.ndarray:
    return np.asarray([sample.label for sample in samples], dtype=float)


def bucket(value: float, thresholds: list[float]) -> float:
    """Return the index `i` of the first threshold satisfying `value < thresholds[i]`.

    Encodes spec [lo, hi) bucket semantics: a value equal to `thresholds[i]`
    belongs to bucket `i+1`, not bucket `i`. Returns `len(thresholds)` for
    values past the last threshold.
    """
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return float(index)
    return float(len(thresholds))


def _numeric_column(prediction_inputs: list[dict], key: str) -> np.ndarray:
    """Extract one raw numeric column under the shared `_numeric` contract."""
    return np.array([_numeric(item.get(key)) for item in prediction_inputs], dtype=float)


def _bucketize(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Vectorized `bucket()`: number of thresholds <= value (equal-to moves up)."""
    return np.searchsorted(thresholds, values, side="right").astype(float)


def _kv_cache_usage_bucket(kv_usage: np.ndarray) -> np.ndarray:
    # astype(int64) truncates toward zero like int(); clip mirrors min(MAX, max(0, .)).
    raw = (kv_usage * KV_CACHE_BUCKET_COUNT).astype(np.int64)
    return np.clip(raw, 0, KV_CACHE_BUCKET_MAX).astype(float)


def _column_length(engineered_columns: dict[str, np.ndarray]) -> int:
    for values in engineered_columns.values():
        return len(values)
    return 0
