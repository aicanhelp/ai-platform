# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema import (
    ALL_FEATURE_COLUMNS,
    MONOTONE_DIRECTION,
    TPOT_FEATURE_COLUMNS,
    TTFT_FEATURE_COLUMNS,
    columns_for_slot,
    monotone_constraints_for,
)
from training.features import feature_matrix_for_inputs, feature_row_for_input
from training.metrics import ranking_metrics, stratified_ranking_metrics

# Raw numeric input fields the feature builder reads. Used by the parity tests to
# generate varied dict inputs that exercise missing / falsey / None handling.
_RAW_NUMERIC_FIELDS = [
    "inputTokenLength",
    "prefixCacheScore",
    "numRequestWaiting",
    "numRequestRunning",
    "kvCacheUsage",
    "inflightTokens",
    "hbmUtilization",
    "aiCoreUtilization",
    "bandwidthUtilization",
]
_INTEGER_FIELDS = {"inputTokenLength", "numRequestWaiting", "numRequestRunning", "inflightTokens"}


def _assert_batch_matches_scalar(inputs: list[dict], columns: list[str] | None) -> None:
    """The batch matrix row-for-row must equal the scalar oracle, exactly.

    The two paths run identical float64 arithmetic, so equality is exact rather
    than tolerance-based; any drift is a real train/serve skew bug, not rounding.
    """
    matrix = feature_matrix_for_inputs(inputs, columns)
    expected_width = len(columns if columns is not None else ALL_FEATURE_COLUMNS)
    assert matrix.shape == (len(inputs), expected_width)
    for row, item in zip(matrix, inputs):
        np.testing.assert_array_equal(row, feature_row_for_input(item, columns))


def test_columns_for_slot_selects_by_target() -> None:
    assert columns_for_slot("aggregate_ttft") == TTFT_FEATURE_COLUMNS
    assert columns_for_slot("disagg_tpot") == TPOT_FEATURE_COLUMNS
    # TPOT drops prefill/prefix-cache-only columns but keeps context length.
    assert "prefixCacheScore" not in TPOT_FEATURE_COLUMNS
    assert "effectiveInputTokens" not in TPOT_FEATURE_COLUMNS
    assert "inputTokenLength" in TPOT_FEATURE_COLUMNS


def test_columns_for_slot_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        columns_for_slot("aggregate_latency")


def test_monotone_constraints_align_to_columns() -> None:
    columns = TTFT_FEATURE_COLUMNS
    constraints = monotone_constraints_for(columns)
    assert len(constraints) == len(columns)
    for column, value in zip(columns, constraints):
        assert value == MONOTONE_DIRECTION.get(column, 0)
    # Strong-semantic directions; load/util features stay unconstrained.
    assert constraints[columns.index("numRequestWaiting")] == 1
    assert constraints[columns.index("prefixCacheScore")] == -1
    assert constraints[columns.index("inflightTokens")] == 0


def test_is_queued_feature_is_derived() -> None:
    idle = feature_row_for_input({"numRequestWaiting": 0}, ["isQueued"])
    queued = feature_row_for_input({"numRequestWaiting": 3}, ["isQueued"])
    assert idle == [0.0]
    assert queued == [1.0]


def test_ranking_metrics_handle_degenerate_inputs() -> None:
    assert ranking_metrics(np.array([1.0]), np.array([1.0]))["valid"] is False
    tied = ranking_metrics(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0]))
    assert tied["valid"] is False


def test_ranking_metrics_reward_correct_order() -> None:
    labels = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = ranking_metrics(labels, labels)
    assert perfect["valid"] is True
    assert perfect["pairwiseAccuracy"] == 1.0
    assert perfect["spearman"] == pytest.approx(1.0)

    reversed_pred = ranking_metrics(labels, labels[::-1])
    assert reversed_pred["pairwiseAccuracy"] == 0.0


def test_stratified_ranking_metrics_split_masks() -> None:
    labels = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    strata = {"low": np.array([True, True, False, False]), "high": np.array([False, False, True, True])}
    result = stratified_ranking_metrics(labels, predictions, strata)
    assert set(result) == {"overall", "low", "high"}
    assert result["low"]["n"] == 2


def test_batch_matches_scalar_on_bucket_boundaries() -> None:
    # Drive prefixCacheScore and numRequestWaiting onto every bucket() boundary so
    # the vectorized searchsorted path is checked at the exact equal-to-threshold
    # values where lo/hi semantics decide the bucket.
    boundaries = [0.0, 0.24, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 11.0, 12.0]
    inputs = [
        {
            "inputTokenLength": 128,
            "prefixCacheScore": min(value, 1.0),
            "numRequestWaiting": value,
            "numRequestRunning": 2,
            "kvCacheUsage": min(value, 1.0),
            "inflightTokens": 10,
            "hbmUtilization": 0.5,
            "aiCoreUtilization": 0.5,
            "bandwidthUtilization": 0.5,
        }
        for value in boundaries
    ]
    _assert_batch_matches_scalar(inputs, None)


def test_batch_matches_scalar_on_kv_cache_edges() -> None:
    # kvCacheUsageBucket truncates toward zero via int(); cover values that sit on
    # the *20 boundaries, the 0.999/1.0 transition, and out-of-[0,1] values that
    # must clamp rather than overflow the 0..19 bucket range.
    values = [-0.5, -0.01, 0.0, 0.0499, 0.05, 0.5, 0.949, 0.95, 0.999, 1.0, 1.05, 2.0]
    inputs = [{"kvCacheUsage": value, "inputTokenLength": 64, "numRequestWaiting": 1} for value in values]
    _assert_batch_matches_scalar(inputs, None)


def test_batch_matches_scalar_on_missing_and_falsey() -> None:
    # One repo-level contract: missing, falsey, and present-None numeric fields all
    # resolve to 0.0 on both paths — including direct columns, which the scalar
    # projection previously crashed on for present-None.
    inputs = [
        {},
        {"inputTokenLength": 0, "kvCacheUsage": 0.0, "numRequestWaiting": 0, "prefixCacheScore": 0.0},
        {
            "inputTokenLength": None,
            "prefixCacheScore": None,
            "numRequestWaiting": None,
            "numRequestRunning": None,
            "kvCacheUsage": None,
            "inflightTokens": None,
            "hbmUtilization": None,
            "aiCoreUtilization": None,
            "bandwidthUtilization": None,
        },
        {"inputTokenLength": "", "prefixCacheScore": "", "kvCacheUsage": ""},
    ]
    _assert_batch_matches_scalar(inputs, None)


def test_present_none_numeric_coerces_to_zero() -> None:
    # Pin the unified contract directly on a direct column (no falsey fallback in
    # the legacy scalar projection) for both the scalar and batch entry points.
    assert feature_row_for_input({"numRequestRunning": None}, ["numRequestRunning"]) == [0.0]
    matrix = feature_matrix_for_inputs([{"numRequestRunning": None}], ["numRequestRunning"])
    np.testing.assert_array_equal(matrix, np.array([[0.0]]))


def test_batch_matches_scalar_across_column_sets() -> None:
    inputs = [
        {
            "inputTokenLength": 200 + index * 7,
            "prefixCacheScore": (index % 5) / 5.0,
            "numRequestWaiting": index % 13,
            "numRequestRunning": index % 4,
            "kvCacheUsage": (index % 21) / 20.0,
            "inflightTokens": index * 3,
            "hbmUtilization": 0.3 + (index % 3) / 10.0,
            "aiCoreUtilization": 0.4,
            "bandwidthUtilization": 0.5,
        }
        for index in range(8)
    ]
    for columns in (None, TTFT_FEATURE_COLUMNS, TPOT_FEATURE_COLUMNS, ["isQueued"], ["kvCacheUsageBucket", "queueBucket"]):
        _assert_batch_matches_scalar(inputs, columns)


def test_batch_empty_inputs_returns_zero_rows() -> None:
    # An empty slot must yield a (0, width) matrix so downstream predict/reshape
    # stays 2-D rather than collapsing to a (0,) vector.
    assert feature_matrix_for_inputs([], TTFT_FEATURE_COLUMNS).shape == (0, len(TTFT_FEATURE_COLUMNS))
    assert feature_matrix_for_inputs([], None).shape == (0, len(ALL_FEATURE_COLUMNS))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [
        "kvCacheUsage",  # int-cast derived source (scalar already raised)
        "numRequestWaiting",  # int-cast derived source
        "prefixCacheScore",  # float derived source (scalar previously leaked nan)
        "hbmUtilization",  # direct-only column (scalar previously leaked nan to model)
    ],
)
def test_non_finite_numeric_rejected_on_both_paths(field: str, bad_value: float) -> None:
    # Proto doubles can carry NaN/inf; both the scalar oracle and the batch path
    # must reject them identically as ValueError (→ INVALID_ARGUMENT) rather than
    # one raising while the other silently buckets or poisons the model.
    item = {field: bad_value}
    with pytest.raises(ValueError):
        feature_row_for_input(item, None)
    with pytest.raises(ValueError):
        feature_matrix_for_inputs([item], None)


def test_batch_matches_scalar_randomized() -> None:
    rng = random.Random(20240612)
    inputs: list[dict] = []
    for _ in range(200):
        item: dict = {}
        for field in _RAW_NUMERIC_FIELDS:
            draw = rng.random()
            if draw < 0.15:
                continue  # field missing entirely
            if draw < 0.2:
                item[field] = None  # present-but-None
            elif field in _INTEGER_FIELDS:
                item[field] = rng.randint(0, 500)
            else:
                item[field] = rng.uniform(-0.2, 1.5)
        inputs.append(item)
    for columns in (None, TTFT_FEATURE_COLUMNS, TPOT_FEATURE_COLUMNS):
        _assert_batch_matches_scalar(inputs, columns)
