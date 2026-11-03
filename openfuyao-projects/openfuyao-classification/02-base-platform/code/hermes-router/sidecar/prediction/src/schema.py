# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared schema constants for prediction runtime and training."""

from __future__ import annotations

JSONL_SCHEMA_VERSION = 1

POD_TYPE_AGGREGATE = "aggregate"
POD_TYPE_PREFILL = "prefill"
POD_TYPE_DECODE = "decode"
POD_TYPES = {POD_TYPE_AGGREGATE, POD_TYPE_PREFILL, POD_TYPE_DECODE}

ROLE_LEADER = "leader"
ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"

TARGET_TTFT = "ttft"
TARGET_TPOT = "tpot"
MODE_AGGREGATE = "aggregate"
MODE_DISAGG = "disagg"

DIRECT_FEATURE_COLUMNS = [
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

DERIVED_FEATURE_COLUMNS = [
    "effectiveInputTokens",
    "prefixCacheBucket",
    "queueBucket",
    "kvCacheUsageBucket",
    # isQueued gives tree models a clean 0/1 split between idle and queued
    # regimes so they need not rediscover that boundary from numRequestWaiting.
    "isQueued",
]

ALL_FEATURE_COLUMNS = DIRECT_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

# TTFT keeps the full feature set. TPOT drops prefill/prefix-cache-only signals:
# per-token decode latency does not depend on the prefix-cache hit, so feeding
# those columns to a TPOT model only adds spurious split dimensions. This mirrors
# the GIE latencypredictor, which gives TTFT and TPOT distinct feature sets.
TPOT_EXCLUDED_COLUMNS = {"prefixCacheScore", "effectiveInputTokens", "prefixCacheBucket"}
TTFT_FEATURE_COLUMNS = list(ALL_FEATURE_COLUMNS)
TPOT_FEATURE_COLUMNS = [column for column in ALL_FEATURE_COLUMNS if column not in TPOT_EXCLUDED_COLUMNS]

# Queue state is now an in-model feature (numRequestWaiting / isQueued) rather
# than a slot dimension, so one model per (mode, target) spans both regimes.
SLOTS = [
    "aggregate_ttft",
    "aggregate_tpot",
    "disagg_ttft",
    "disagg_tpot",
]

# Monotone direction for strong-semantic features only (+1 increasing, -1
# decreasing). Load/utilization features (numRequestRunning, inflightTokens,
# *Utilization, kvCacheUsage*) are intentionally left unconstrained: batching and
# cache-locality effects can make their local relationship with latency
# non-monotonic, so a hard constraint there would hurt rather than denoise.
MONOTONE_DIRECTION = {
    "numRequestWaiting": 1,
    "isQueued": 1,
    "queueBucket": 1,
    "inputTokenLength": 1,
    "effectiveInputTokens": 1,
    "prefixCacheScore": -1,
    "prefixCacheBucket": -1,
}


def slot_name(mode: str, target: str) -> str:
    return f"{mode}_{target}"


def columns_for_slot(slot: str) -> list[str]:
    """Return the feature columns a slot's model is trained and served on."""
    if slot.endswith(f"_{TARGET_TPOT}"):
        return TPOT_FEATURE_COLUMNS
    if slot.endswith(f"_{TARGET_TTFT}"):
        return TTFT_FEATURE_COLUMNS
    raise ValueError(f"unknown slot {slot!r}")


def monotone_constraints_for(columns: list[str]) -> tuple[int, ...]:
    """Per-column monotone constraints aligned to `columns` order (0 = free)."""
    return tuple(MONOTONE_DIRECTION.get(column, 0) for column in columns)
