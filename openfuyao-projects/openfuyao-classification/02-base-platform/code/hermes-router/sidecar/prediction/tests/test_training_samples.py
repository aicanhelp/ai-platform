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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.jsonl_loader import load_completed_jsonl
from training.samples import generate_samples


def test_completed_jsonl_generates_expected_samples() -> None:
    records, load_drops = load_completed_jsonl(ROOT / "tests" / "fixtures" / "completed_requests.jsonl")
    samples, sample_drops = generate_samples(records)

    assert len(records) == 12
    assert load_drops == {}
    assert len(samples) == 24
    # Queue state is an in-model feature now, so slots collapse to one per
    # (mode, target); queued and idle samples share a slot.
    assert {sample.slot for sample in samples} == {
        "aggregate_ttft",
        "aggregate_tpot",
        "disagg_ttft",
        "disagg_tpot",
    }
    assert sample_drops == {}


def test_loader_counts_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"requestId":"missing-schema"}',
                '{"schemaVersion":99,"requestId":"bad-version"}',
                '{"schemaVersion":1,"requestId":"missing-completed","model":"m","predictionInputs":{"leader":{}}}',
                '{"schemaVersion":1,"requestId":"missing-model","completedAt":"2026-05-20T00:00:00Z","predictionInputs":{"leader":{}}}',
            ]
        ),
        encoding="utf-8",
    )

    records, drops = load_completed_jsonl(path)

    assert records == []
    assert drops["schema_version_missing"] == 1
    assert drops["schema_version_unknown"] == 1
    assert drops["completed_at_missing"] == 1
    assert drops["model_missing"] == 1
