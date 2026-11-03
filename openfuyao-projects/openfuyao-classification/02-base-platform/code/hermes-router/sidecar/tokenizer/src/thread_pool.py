# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Thread-pool sizing for tokenizer CPU work."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

THREAD_POOL_SIZE_ENV = "TOKENIZER_THREAD_POOL_SIZE"
THREAD_POOL_CPU_MULTIPLIER = 2
MAX_DEFAULT_THREAD_POOL_WORKERS = 32


def available_cpu_count() -> int:
    with contextlib.suppress(OSError, ValueError):
        quota, period = _read_cgroup_v2_cpu_quota()
        if quota > 0 and period > 0:
            return max(1, int(quota / period))

    with contextlib.suppress(OSError, ValueError):
        quota, period = _read_cgroup_v1_cpu_quota()
        if quota > 0 and period > 0:
            return max(1, int(quota / period))

    with contextlib.suppress(AttributeError, OSError):
        available = len(os.sched_getaffinity(0))
        if available > 0:
            return available

    return os.cpu_count() or 1


def resolve_thread_pool_size(
    env: Mapping[str, str] | None = None,
    override: int | None = None,
) -> int:
    """Resolve the worker count with precedence: explicit override > env > default.

    The override is the CLI ``--thread-pool-size`` value; the env var is
    ``TOKENIZER_THREAD_POOL_SIZE``; the default scales with detected CPUs.
    """
    if override is not None:
        if override < 1:
            raise ValueError("thread pool size must be at least 1")
        return override

    env = env or os.environ
    configured = env.get(THREAD_POOL_SIZE_ENV)
    if configured:
        try:
            workers = int(configured)
        except ValueError as exc:
            raise ValueError(f"{THREAD_POOL_SIZE_ENV} must be an integer") from exc
        if workers < 1:
            raise ValueError(f"{THREAD_POOL_SIZE_ENV} must be at least 1")
        return workers

    return min(
        available_cpu_count() * THREAD_POOL_CPU_MULTIPLIER,
        MAX_DEFAULT_THREAD_POOL_WORKERS,
    )


def create_thread_pool(max_workers: int | None = None) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=resolve_thread_pool_size(override=max_workers),
        thread_name_prefix="tokenizer",
    )


def _read_cgroup_v2_cpu_quota() -> tuple[int, int]:
    quota_text, period_text = _read_text("/sys/fs/cgroup/cpu.max").split()
    if quota_text == "max":
        return -1, int(period_text)
    return int(quota_text), int(period_text)


def _read_cgroup_v1_cpu_quota() -> tuple[int, int]:
    return (
        int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")),
        int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")),
    )


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read().strip()