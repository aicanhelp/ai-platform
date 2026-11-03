# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Dataclasses shared by runtime and trainer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PredictInput:
    input_id: int
    pod_type: str
    input_token_length: int
    prefix_cache_score: float
    num_request_waiting: int
    num_request_running: int
    kv_cache_usage: float
    inflight_tokens: int
    hbm_utilization: float
    ai_core_utilization: float
    bandwidth_utilization: float
    server_ip: str


@dataclass(frozen=True)
class PredictOutput:
    input_id: int
    server_ip: str
    predicted_ttft_millis: float
    predicted_tpot_millis: float


@dataclass(frozen=True)
class CompletedRecord:
    append_order: int
    raw: dict[str, Any]
    schema_version: int
    completed_at: datetime
    model: str
    ttft_millis: float | None
    total_latency_millis: float | None
    output_length: int | None
    prediction_inputs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TrainingSample:
    target_model: str
    slot: str
    target: str
    label: float
    completed_at: datetime
    append_order: int
    prediction_input: dict[str, Any]
