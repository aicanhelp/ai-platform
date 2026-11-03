# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Training sample generation from completed records."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from domain.models import CompletedRecord, TrainingSample
from schema import (
    DIRECT_FEATURE_COLUMNS,
    MODE_AGGREGATE,
    MODE_DISAGG,
    POD_TYPES,
    ROLE_DECODE,
    ROLE_LEADER,
    ROLE_PREFILL,
    TARGET_TPOT,
    TARGET_TTFT,
    slot_name,
)


def generate_samples(records: list[CompletedRecord]) -> tuple[list[TrainingSample], Counter[str]]:
    samples: list[TrainingSample] = []
    drops: Counter[str] = Counter()
    for record in records:
        # Each role's presence drives its own sample emission so a PD record
        # missing the complementary role is not counted as a spurious drop.
        if ROLE_LEADER in record.prediction_inputs:
            emit_ttft(record, ROLE_LEADER, MODE_AGGREGATE, samples, drops)
            emit_tpot(record, ROLE_LEADER, MODE_AGGREGATE, samples, drops)
        if ROLE_PREFILL in record.prediction_inputs:
            emit_ttft(record, ROLE_PREFILL, MODE_DISAGG, samples, drops)
        if ROLE_DECODE in record.prediction_inputs:
            emit_tpot(record, ROLE_DECODE, MODE_DISAGG, samples, drops)
    return samples, drops


def emit_ttft(
    record: CompletedRecord,
    role: str,
    mode: str,
    samples: list[TrainingSample],
    drops: Counter[str],
) -> None:
    prediction_input = valid_prediction_input(record, role, drops)
    if prediction_input is None:
        return
    if record.ttft_millis is None or record.ttft_millis <= 0:
        drops["ttft_label_invalid"] += 1
        return
    samples.append(
        TrainingSample(
            target_model=record.model,
            slot=slot_name(mode, TARGET_TTFT),
            target=TARGET_TTFT,
            label=record.ttft_millis,
            completed_at=record.completed_at,
            append_order=record.append_order,
            prediction_input=prediction_input,
        )
    )


def emit_tpot(
    record: CompletedRecord,
    role: str,
    mode: str,
    samples: list[TrainingSample],
    drops: Counter[str],
) -> None:
    prediction_input = valid_prediction_input(record, role, drops)
    if prediction_input is None:
        return
    if (
        record.output_length is None
        or record.output_length <= 1
        or record.total_latency_millis is None
        or record.ttft_millis is None
        or record.total_latency_millis <= record.ttft_millis
    ):
        drops["tpot_label_invalid"] += 1
        return
    label = (record.total_latency_millis - record.ttft_millis) / (record.output_length - 1)
    samples.append(
        TrainingSample(
            target_model=record.model,
            slot=slot_name(mode, TARGET_TPOT),
            target=TARGET_TPOT,
            label=label,
            completed_at=record.completed_at,
            append_order=record.append_order,
            prediction_input=prediction_input,
        )
    )


def valid_prediction_input(
    record: CompletedRecord,
    role: str,
    drops: Counter[str],
) -> dict[str, Any] | None:
    value = record.prediction_inputs.get(role)
    if not isinstance(value, dict):
        drops["prediction_inputs_missing"] += 1
        return None
    if value.get("podType") not in POD_TYPES:
        drops["pod_type_invalid"] += 1
        return None
    if not positive_number(value.get("inputTokenLength")):
        drops["input_token_length_invalid"] += 1
        return None
    for column in DIRECT_FEATURE_COLUMNS:
        if not finite_number(value.get(column)):
            drops["numeric_field_non_finite"] += 1
            return None
    return value


def positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0


def finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))
