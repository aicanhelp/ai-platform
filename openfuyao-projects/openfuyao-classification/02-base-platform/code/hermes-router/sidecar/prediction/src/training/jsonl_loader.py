# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Completed-request JSONL loader."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from domain.models import CompletedRecord
from schema import JSONL_SCHEMA_VERSION


def load_completed_jsonl(path: Path) -> tuple[list[CompletedRecord], Counter[str]]:
    records: list[CompletedRecord] = []
    drops: Counter[str] = Counter()
    append_order = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                drops["json_invalid"] += 1
                continue
            record = parse_record(raw, append_order, drops)
            append_order += 1
            if record is not None:
                records.append(record)
    return records, drops


def parse_record(raw: dict[str, Any], append_order: int, drops: Counter[str]) -> CompletedRecord | None:
    if "schemaVersion" not in raw:
        drops["schema_version_missing"] += 1
        return None
    if raw.get("schemaVersion") != JSONL_SCHEMA_VERSION:
        drops["schema_version_unknown"] += 1
        return None
    completed_at_raw = raw.get("completedAt")
    if not completed_at_raw:
        drops["completed_at_missing"] += 1
        return None
    completed_at = parse_datetime(completed_at_raw)
    if completed_at is None:
        drops["completed_at_missing"] += 1
        return None
    model = str(raw.get("model") or "")
    if not model:
        drops["model_missing"] += 1
        return None
    prediction_inputs = raw.get("predictionInputs")
    if not isinstance(prediction_inputs, dict) or not prediction_inputs:
        drops["prediction_inputs_missing"] += 1
        return None
    return CompletedRecord(
        append_order=append_order,
        raw=raw,
        schema_version=int(raw["schemaVersion"]),
        completed_at=completed_at,
        model=model,
        ttft_millis=number_or_none(raw.get("ttftMillis")),
        total_latency_millis=number_or_none(raw.get("totalLatencyMillis")),
        output_length=int(raw["outputLength"]) if isinstance(raw.get("outputLength"), int | float) else None,
        prediction_inputs=prediction_inputs,
    )


def parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def number_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
