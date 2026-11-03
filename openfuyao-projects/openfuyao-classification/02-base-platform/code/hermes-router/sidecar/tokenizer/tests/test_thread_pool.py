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

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import thread_pool


def test_resolve_thread_pool_size_uses_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(thread_pool.THREAD_POOL_SIZE_ENV, "7")

    assert thread_pool.resolve_thread_pool_size() == 7


def test_resolve_thread_pool_size_uses_cpu_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(thread_pool.THREAD_POOL_SIZE_ENV, raising=False)
    monkeypatch.setattr(thread_pool, "available_cpu_count", lambda: 4)

    assert thread_pool.resolve_thread_pool_size() == 8


def test_resolve_thread_pool_size_caps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(thread_pool.THREAD_POOL_SIZE_ENV, raising=False)
    monkeypatch.setattr(thread_pool, "available_cpu_count", lambda: 64)

    assert thread_pool.resolve_thread_pool_size() == thread_pool.MAX_DEFAULT_THREAD_POOL_WORKERS


def test_resolve_thread_pool_size_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(thread_pool.THREAD_POOL_SIZE_ENV, "0")

    with pytest.raises(ValueError, match=thread_pool.THREAD_POOL_SIZE_ENV):
        thread_pool.resolve_thread_pool_size()


def test_resolve_thread_pool_size_explicit_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(thread_pool.THREAD_POOL_SIZE_ENV, "7")

    assert thread_pool.resolve_thread_pool_size(override=3) == 3


def test_resolve_thread_pool_size_rejects_invalid_explicit_override() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        thread_pool.resolve_thread_pool_size(override=0)