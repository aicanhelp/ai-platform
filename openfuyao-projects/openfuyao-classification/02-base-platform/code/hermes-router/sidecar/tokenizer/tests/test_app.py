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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "proto"))

import app


class _FakeLoop:
    def __init__(self) -> None:
        self.default_executor: ThreadPoolExecutor | None = None

    def set_default_executor(self, executor: ThreadPoolExecutor) -> None:
        self.default_executor = executor


def test_configure_default_executor_installs_tokenizer_thread_pool(monkeypatch) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    captured: dict[str, int | None] = {}

    def fake_create_thread_pool(max_workers: int | None = None) -> ThreadPoolExecutor:
        captured["max_workers"] = max_workers
        return executor

    monkeypatch.setattr(app, "create_thread_pool", fake_create_thread_pool)
    loop = _FakeLoop()

    configured = app.configure_default_executor(loop, 5)

    assert configured is executor
    assert loop.default_executor is executor
    assert captured["max_workers"] == 5
    executor.shutdown(wait=True)


def test_parse_args_defaults_to_kserve_aligned_paths(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tokenizer-sidecar"])

    args = app.parse_args()

    # The kserve scheduler preset passes no args to the tokenizer container, so
    # the image defaults are the deployment contract: UDS under the preset's
    # /tmp/tokenizer mount and /healthz on the probe port 8082.
    assert args.socket_path == "/tmp/tokenizer/tokenizer-uds.socket"
    assert args.health_port == 8082