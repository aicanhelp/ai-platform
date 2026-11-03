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

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain.models import PredictInput, PredictOutput
from runtime.artifact_predictor import ArtifactPredictor, input_to_dict
from schema import (
    MODE_AGGREGATE,
    MODE_DISAGG,
    POD_TYPE_AGGREGATE,
    POD_TYPE_DECODE,
    POD_TYPE_PREFILL,
    TARGET_TPOT,
    TARGET_TTFT,
    slot_name,
)
from training.artifacts import TrainOptions, write_artifacts
from training.features import feature_row_for_input, matrix_for_samples
from training.jsonl_loader import load_completed_jsonl
from training.samples import generate_samples
from training.split import temporal_holdout


def load_fixture_samples():
    records, load_drops = load_completed_jsonl(ROOT / "tests" / "fixtures" / "completed_requests.jsonl")
    samples, sample_drops = generate_samples(records)
    drops = Counter()
    drops.update(load_drops)
    drops.update(sample_drops)
    return samples, drops


def test_feature_engineering_and_temporal_split_are_deterministic() -> None:
    samples, _ = load_fixture_samples()
    slot_samples = [sample for sample in samples if sample.slot == "aggregate_ttft"]

    train, holdout = temporal_holdout(slot_samples, 0.2)
    matrix = matrix_for_samples(train)

    assert len(train) == 4
    assert len(holdout) == 2
    assert matrix.shape[0] == 4
    assert holdout[0].append_order > train[-1].append_order


def test_xgboost_artifact_round_trip(tmp_path: Path) -> None:
    bundle = train_bundle(tmp_path, "xgboost", "xgb-v1")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "xgboost"
    assert len(manifest["promotedSlots"]) == 4

    predictor = ArtifactPredictor(tmp_path, "test-model", "xgb-v1")
    outputs = predictor.predict(
        "test-model",
        [
            PredictInput(0, "aggregate", 120, 0.2, 0, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.1"),
            PredictInput(1, "prefill", 100, 0.4, 1, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.2"),
            PredictInput(2, "decode", 100, 0.0, 1, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.3"),
        ],
    )

    assert [output.input_id for output in outputs] == [0, 1, 2]
    assert outputs[0].predicted_ttft_millis >= 0
    assert outputs[0].predicted_tpot_millis >= 0


def test_lightgbm_artifact_round_trip(tmp_path: Path) -> None:
    bundle = train_bundle(tmp_path, "lightgbm", "lgbm-v1")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "lightgbm"
    assert len(manifest["promotedSlots"]) == 4

    predictor = ArtifactPredictor(tmp_path, "test-model", "lgbm-v1")
    outputs = predictor.predict(
        "test-model",
        [PredictInput(0, "aggregate", 120, 0.2, 2, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.1")],
    )

    assert len(outputs) == 1
    assert outputs[0].predicted_ttft_millis >= 0


def test_bundle_exposes_per_slot_columns_ranking_and_monotone(tmp_path: Path) -> None:
    bundle = train_bundle(tmp_path, "xgboost", "xgb-contract")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((bundle / "report.json").read_text(encoding="utf-8"))

    assert manifest["bundleVersion"] == 2
    # TPOT models drop prefill/prefix-cache-only columns; TTFT keeps them.
    tpot_columns = manifest["slotFeatureColumns"]["aggregate_tpot"]
    assert "prefixCacheScore" not in tpot_columns
    assert "effectiveInputTokens" not in tpot_columns
    assert "prefixCacheBucket" not in tpot_columns
    assert "isQueued" in manifest["slotFeatureColumns"]["aggregate_ttft"]

    # Monotone constraints are on by default and recorded for traceability.
    assert metadata["monotoneConstraintsEnabled"] is True
    assert len(metadata["slotMonotoneConstraints"]["aggregate_ttft"]) == len(
        manifest["slotFeatureColumns"]["aggregate_ttft"]
    )

    ranking = report["slots"]["aggregate_ttft"]["ranking"]
    assert "overall" in ranking
    assert "queued" in ranking and "noQueue" in ranking


def _mixed_inputs() -> list[PredictInput]:
    # 12 backends spanning every pod type, exercising the batched slot grouping.
    inputs: list[PredictInput] = []
    for index in range(12):
        pod_type = (POD_TYPE_AGGREGATE, POD_TYPE_PREFILL, POD_TYPE_DECODE)[index % 3]
        inputs.append(
            PredictInput(
                input_id=index,
                pod_type=pod_type,
                input_token_length=80 + index * 11,
                prefix_cache_score=(index % 5) / 5.0,
                num_request_waiting=index % 4,
                num_request_running=index % 3,
                kv_cache_usage=(index % 7) / 10.0,
                inflight_tokens=index * 3,
                hbm_utilization=0.3 + (index % 3) / 10.0,
                ai_core_utilization=0.4,
                bandwidth_utilization=0.5,
                server_ip=f"10.0.9.{index}",
            )
        )
    return inputs


def _reference_per_row(predictor: ArtifactPredictor, inputs: list[PredictInput]) -> list[PredictOutput]:
    """Golden reimplementation of the legacy per-row predict path.

    Mirrors the original one-row-at-a-time implementation the batched rewrite
    replaced, per backend: XGBoost wrapped each row in a DMatrix, LightGBM called
    predict on a single-row matrix directly. Used to prove the batched path is
    numerically equivalent for both backends.
    """

    def predict_one(slot: str, prediction_input: dict) -> float:
        row = feature_row_for_input(prediction_input, predictor._columns_for(slot))
        matrix = np.asarray([row], dtype=float)
        if predictor.backend == "xgboost":
            import xgboost as xgb

            return float(predictor.models[slot].predict(xgb.DMatrix(matrix))[0])
        return float(predictor.models[slot].predict(matrix)[0])

    outputs: list[PredictOutput] = []
    for item in inputs:
        prediction_input = input_to_dict(item)
        if item.pod_type == POD_TYPE_AGGREGATE:
            ttft_slot = slot_name(MODE_AGGREGATE, TARGET_TTFT)
            tpot_slot = slot_name(MODE_AGGREGATE, TARGET_TPOT)
            if ttft_slot not in predictor.models or tpot_slot not in predictor.models:
                continue
            ttft, tpot = predict_one(ttft_slot, prediction_input), predict_one(tpot_slot, prediction_input)
        elif item.pod_type == POD_TYPE_PREFILL:
            ttft_slot = slot_name(MODE_DISAGG, TARGET_TTFT)
            if ttft_slot not in predictor.models:
                continue
            ttft, tpot = predict_one(ttft_slot, prediction_input), 0.0
        elif item.pod_type == POD_TYPE_DECODE:
            tpot_slot = slot_name(MODE_DISAGG, TARGET_TPOT)
            if tpot_slot not in predictor.models:
                continue
            ttft, tpot = 0.0, predict_one(tpot_slot, prediction_input)
        else:
            continue
        outputs.append(
            PredictOutput(item.input_id, item.server_ip, max(ttft, 0.0), max(tpot, 0.0))
        )
    return outputs


@pytest.mark.parametrize(
    ("backend", "version"),
    [("xgboost", "xgb-equiv"), ("lightgbm", "lgbm-equiv")],
)
def test_batched_predict_matches_per_row_reference(tmp_path: Path, backend: str, version: str) -> None:
    train_bundle(tmp_path, backend, version)
    predictor = ArtifactPredictor(tmp_path, "test-model", version)
    inputs = _mixed_inputs()

    outputs = predictor.predict("test-model", inputs)
    reference = _reference_per_row(predictor, inputs)

    assert [o.input_id for o in outputs] == [o.input_id for o in reference]
    for got, want in zip(outputs, reference):
        # inplace_predict vs DMatrix path: equal up to float32 inference noise.
        assert got.predicted_ttft_millis == pytest.approx(want.predicted_ttft_millis, abs=1e-4)
        assert got.predicted_tpot_millis == pytest.approx(want.predicted_tpot_millis, abs=1e-4)


def test_predict_preserves_order_and_omits_unknown_pod_type(tmp_path: Path) -> None:
    train_bundle(tmp_path, "xgboost", "xgb-order")
    predictor = ArtifactPredictor(tmp_path, "test-model", "xgb-order")

    inputs = [
        PredictInput(0, "aggregate", 120, 0.2, 0, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.0"),
        PredictInput(1, "bogus", 100, 0.4, 1, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.1"),
        PredictInput(2, "decode", 100, 0.0, 1, 2, 0.3, 12, 0.4, 0.5, 0.6, "10.0.9.2"),
    ]
    outputs = predictor.predict("test-model", inputs)

    # Unknown pod type is dropped; surviving outputs keep input order.
    assert [o.input_id for o in outputs] == [0, 2]
    # decode has no prefill phase, so its TTFT stays the fixed 0.0.
    assert outputs[1].predicted_ttft_millis == 0.0


def train_bundle(tmp_path: Path, backend: str, version: str) -> Path:
    samples, drops = load_fixture_samples()
    # Fixture data has 2-3 samples per slot; override the spec-baseline
    # regularization so trees can still fit. Production defaults remain in the
    # backend modules — these overrides exist only to keep the test fast.
    if backend == "xgboost":
        hyperparams = {"n_estimators": 25, "max_depth": 3, "min_child_weight": 1, "gamma": 0.0}
    else:
        hyperparams = {"n_estimators": 25, "max_depth": 3, "min_child_samples": 1}
    return write_artifacts(
        samples,
        drops,
        TrainOptions(
            artifact_root=tmp_path,
            target_model="test-model",
            model_version=version,
            model_backend=backend,
            quantile=0.9,
            min_samples_per_slot=2,
            min_holdout_samples_per_slot=1,
            holdout_ratio=0.2,
            bucket_cap=0,
            hyperparams=hyperparams,
        ),
    )
