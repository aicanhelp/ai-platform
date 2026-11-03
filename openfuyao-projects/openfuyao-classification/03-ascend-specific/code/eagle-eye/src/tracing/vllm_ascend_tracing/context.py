# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Global state and utilities for vLLM-Ascend tracing."""

import threading
from typing import Dict, Optional

from opentelemetry import trace

# Global dictionary to store active request spans
_request_spans: Dict[str, trace.Span] = {}
_lock = threading.Lock()


def store_span(span_id: str, span: trace.Span) -> None:
    with _lock:
        _request_spans[span_id] = span


def pop_span(span_id: str) -> Optional[trace.Span]:
    with _lock:
        return _request_spans.pop(span_id, None)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("vllm_ascend.tracing")
