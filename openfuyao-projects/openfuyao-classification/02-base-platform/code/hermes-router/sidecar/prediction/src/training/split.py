# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Deterministic sample splitting and bucket capping."""

from __future__ import annotations

import math
from collections import defaultdict

from domain.models import TrainingSample


def temporal_holdout(
    samples: list[TrainingSample],
    holdout_ratio: float,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    ordered = sorted(samples, key=lambda item: (item.completed_at, item.append_order))
    if len(ordered) <= 1:
        return ordered, []
    holdout_count = max(1, int(math.ceil(len(ordered) * holdout_ratio)))
    holdout_count = min(holdout_count, len(ordered) - 1)
    return ordered[:-holdout_count], ordered[-holdout_count:]


def capped_bucket_sample(samples: list[TrainingSample], bucket_cap: int) -> list[TrainingSample]:
    if bucket_cap <= 0:
        return samples
    buckets: dict[tuple[str, int], list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        waiting = int(sample.prediction_input.get("numRequestWaiting") or 0)
        buckets[(sample.slot, waiting)].append(sample)
    capped: list[TrainingSample] = []
    for key in sorted(buckets):
        # Keep the most recent samples per bucket rather than the oldest, so a
        # long-spanning trace does not bias each bucket toward stale data.
        capped.extend(buckets[key][-bucket_cap:])
    return sorted(capped, key=lambda item: (item.completed_at, item.append_order))
