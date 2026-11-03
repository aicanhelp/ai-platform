# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Artifact-backed prediction runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from domain.models import PredictInput, PredictOutput
from runtime.predictor import FailedPreconditionError
from schema import (
    MODE_AGGREGATE,
    MODE_DISAGG,
    POD_TYPE_AGGREGATE,
    POD_TYPE_DECODE,
    POD_TYPE_PREFILL,
    TARGET_TPOT,
    TARGET_TTFT,
    columns_for_slot,
    slot_name,
)
from training.features import engineered_feature_columns, project_columns_matrix

# Pod types whose missing slot side carries a fixed 0.0 prediction (prefill has
# no decode phase, decode has no prefill phase). A None slot in the routing table
# means "do not run a model, emit 0.0".
_FIXED_ZERO = 0.0


@dataclass(frozen=True)
class _RowPlan:
    """Where a single backend's feature rows live inside the per-slot batches.

    `*_index` is the row offset into that slot's batched matrix; a None slot
    means the output side is the fixed 0.0 of a disaggregated pod.
    """

    ttft_slot: str | None
    ttft_index: int
    tpot_slot: str | None
    tpot_index: int


class ArtifactPredictor:
    def __init__(self, artifact_root: Path, target_model: str, model_version: str) -> None:
        if not artifact_root:
            raise ValueError("artifact_root is required")
        if not target_model:
            raise ValueError("target_model is required")
        if not model_version:
            raise ValueError("model_version is required")
        self.bundle_dir = artifact_root / target_model / model_version
        self.manifest = read_json(self.bundle_dir / "manifest.json")
        self.metadata = read_json(self.bundle_dir / "metadata.json")
        self.target_model = str(self.manifest.get("targetModel") or "")
        self.model_version = str(self.manifest.get("modelVersion") or "")
        if self.target_model != target_model:
            raise ValueError(f"manifest targetModel {self.target_model!r} does not match {target_model!r}")
        if self.model_version != model_version:
            raise ValueError(f"manifest modelVersion {self.model_version!r} does not match {model_version!r}")
        # Distinguish "field missing" from "fields disagree" so the load failure
        # log line tells the operator exactly which file is broken.
        manifest_backend = self.manifest.get("backend")
        metadata_backend = self.metadata.get("backend")
        if not manifest_backend:
            raise ValueError("manifest.json is missing the 'backend' field")
        if not metadata_backend:
            raise ValueError("metadata.json is missing the 'backend' field")
        if manifest_backend != metadata_backend:
            raise ValueError(
                f"manifest backend {manifest_backend!r} does not match metadata backend {metadata_backend!r}"
            )
        self.backend = str(manifest_backend)
        if self.backend not in {"xgboost", "lightgbm"}:
            raise ValueError(f"unsupported backend {self.backend!r}")
        # Per-slot feature columns: TTFT and TPOT models are trained on different
        # column sets, so the runtime must build each slot's row from its own list.
        self.slot_feature_columns = dict(self.manifest.get("slotFeatureColumns") or {})
        self.models = self._load_models()
        # Precompute hot-path lookups once at load time so a Predict RPC carrying
        # N backends does not re-derive routing or column lists N times.
        self._pod_routing = self._build_pod_routing()
        self._slot_columns = {slot: self._columns_for(slot) for slot in self.models}

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        if target_model != self.target_model:
            raise FailedPreconditionError(
                f"request targetModel {target_model!r} does not match loaded artifact targetModel {self.target_model!r}"
            )
        # Batch every backend that needs the same slot into one matrix and run a
        # single model.predict per slot, instead of one (DMatrix + predict) per
        # backend per slot. For N backends this collapses up to 2N model calls
        # into at most len(SLOTS) calls.
        plans, slot_matrices = self._build_slot_matrices(inputs)
        slot_scores = {slot: self._predict_matrix(slot, matrix) for slot, matrix in slot_matrices.items()}
        return self._assemble_outputs(inputs, plans, slot_scores)

    def _build_slot_matrices(
        self, inputs: list[PredictInput]
    ) -> tuple[list[_RowPlan | None], dict[str, np.ndarray]]:
        # Group surviving inputs by pod type, then vectorize feature construction
        # once per group. Each slot is fed by exactly one pod type, and an
        # aggregate pod's two slots share its group's row order, so engineering a
        # group once and projecting it onto each slot preserves the scalar path's
        # "engineer once, project twice" behavior without re-running bucket math.
        group_inputs: dict[str, list[dict[str, Any]]] = {}
        plans = [self._plan_backend(item, group_inputs) for item in inputs]
        return plans, self._project_groups(group_inputs)

    def _plan_backend(
        self, item: PredictInput, group_inputs: dict[str, list[dict[str, Any]]]
    ) -> _RowPlan | None:
        ttft_slot, tpot_slot = self._pod_routing.get(item.pod_type, (None, None))
        required = [slot for slot in (ttft_slot, tpot_slot) if slot is not None]
        # Omit the backend when its pod type is unknown or any model it needs is
        # not promoted in this bundle — matches the legacy per-item None skip.
        if not required or any(slot not in self.models for slot in required):
            return None
        rows = group_inputs.setdefault(item.pod_type, [])
        row_index = len(rows)
        rows.append(input_to_dict(item))
        # Both slots of a pod type draw from the same group in append order, so a
        # single row index addresses this input in either slot matrix.
        return _RowPlan(
            ttft_slot=ttft_slot,
            ttft_index=row_index if ttft_slot is not None else -1,
            tpot_slot=tpot_slot,
            tpot_index=row_index if tpot_slot is not None else -1,
        )

    def _project_groups(self, group_inputs: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
        slot_matrices: dict[str, np.ndarray] = {}
        for pod_type, rows in group_inputs.items():
            ttft_slot, tpot_slot = self._pod_routing[pod_type]
            engineered = engineered_feature_columns(rows)
            for slot in (ttft_slot, tpot_slot):
                if slot is not None:
                    slot_matrices[slot] = project_columns_matrix(engineered, self._slot_columns[slot])
        return slot_matrices

    def _predict_matrix(self, slot: str, matrix: np.ndarray) -> np.ndarray:
        model = self.models[slot]
        if self.backend == "xgboost":
            # inplace_predict consumes the ndarray directly, skipping the
            # per-call DMatrix construction that dominates small-batch latency.
            preds = model.inplace_predict(matrix)
        else:
            preds = model.predict(matrix)
        # Normalize shape: single-row and multi-row backends can return
        # differently shaped arrays; callers index a flat 1-D vector.
        return np.asarray(preds, dtype=float).reshape(-1)

    def _assemble_outputs(
        self,
        inputs: list[PredictInput],
        plans: list[_RowPlan | None],
        slot_scores: dict[str, np.ndarray],
    ) -> list[PredictOutput]:
        outputs: list[PredictOutput] = []
        for item, plan in zip(inputs, plans):
            if plan is None:
                continue
            ttft = slot_scores[plan.ttft_slot][plan.ttft_index] if plan.ttft_slot else _FIXED_ZERO
            tpot = slot_scores[plan.tpot_slot][plan.tpot_index] if plan.tpot_slot else _FIXED_ZERO
            outputs.append(
                PredictOutput(
                    input_id=item.input_id,
                    server_ip=item.server_ip,
                    predicted_ttft_millis=max(float(ttft), 0.0),
                    predicted_tpot_millis=max(float(tpot), 0.0),
                )
            )
        return outputs

    def _build_pod_routing(self) -> dict[str, tuple[str | None, str | None]]:
        """Map each pod type to its (ttft_slot, tpot_slot); None means fixed 0.0."""
        return {
            POD_TYPE_AGGREGATE: (
                slot_name(MODE_AGGREGATE, TARGET_TTFT),
                slot_name(MODE_AGGREGATE, TARGET_TPOT),
            ),
            POD_TYPE_PREFILL: (slot_name(MODE_DISAGG, TARGET_TTFT), None),
            POD_TYPE_DECODE: (None, slot_name(MODE_DISAGG, TARGET_TPOT)),
        }

    def _load_models(self) -> dict[str, Any]:
        promoted = self.manifest.get("promotedSlots") or []
        slot_files = self.manifest.get("slotFiles") or {}
        models: dict[str, Any] = {}
        for slot in promoted:
            relative_path = slot_files.get(slot)
            if not relative_path:
                raise ValueError(f"promoted slot {slot!r} is missing a slot file entry in manifest.json")
            model_path = self.bundle_dir / relative_path
            # Pre-check the file before handing it to the backend so the
            # "file missing" failure carries the slot name and full path in
            # the load log rather than the backend's generic IOError.
            if not model_path.is_file():
                raise ValueError(
                    f"promoted slot {slot!r} backend file does not exist on disk: {model_path}"
                )
            if self.backend == "xgboost":
                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(model_path))
                models[slot] = booster
            else:
                import lightgbm as lgb

                models[slot] = lgb.Booster(model_file=str(model_path))
        return models

    def _columns_for(self, slot: str) -> list[str]:
        return self.slot_feature_columns.get(slot) or columns_for_slot(slot)


def input_to_dict(item: PredictInput) -> dict[str, Any]:
    return {
        "inputId": item.input_id,
        "podType": item.pod_type,
        "inputTokenLength": item.input_token_length,
        "prefixCacheScore": item.prefix_cache_score,
        "numRequestWaiting": item.num_request_waiting,
        "numRequestRunning": item.num_request_running,
        "kvCacheUsage": item.kv_cache_usage,
        "inflightTokens": item.inflight_tokens,
        "hbmUtilization": item.hbm_utilization,
        "aiCoreUtilization": item.ai_core_utilization,
        "bandwidthUtilization": item.bandwidth_utilization,
        "serverIP": item.server_ip,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
