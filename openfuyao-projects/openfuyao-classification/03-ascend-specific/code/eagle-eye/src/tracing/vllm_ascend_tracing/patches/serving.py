# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patch for OpenAI serving endpoints.

Creates the root span `vllm_ascend.request` and updates trace_headers
so downstream components (EngineCore) receive the new span context.

Uses contextvars to safely pass span info across concurrent async requests,
avoiding race conditions from monkey-patching instance methods.

Covers:
- /v1/chat/completions (OpenAIServingChat.create_chat_completion)
- /v1/completions (OpenAIServingCompletion.create_completion)
- /v1/responses (OpenAIServingResponses.create_responses)
"""

import contextvars
import functools
import logging

from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from vllm_ascend_tracing.context import store_span, pop_span, get_tracer

logger = logging.getLogger(__name__)

_propagator = TraceContextTextMapPropagator()

# ContextVar to hold current request's span info per-coroutine
_current_span_info: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    '_current_span_info', default=None
)


def _wrap_create_method(original_method):
    """Wrapper that creates a root span and stores it in ContextVar."""

    @functools.wraps(original_method)
    async def wrapped(self, request, raw_request=None, **kwargs):
        # Extract parent context from incoming traceparent header
        parent_ctx = None
        if raw_request is not None:
            headers = dict(raw_request.headers)
            parent_ctx = _propagator.extract(carrier=headers)

        tracer = get_tracer()

        # Create root span with parent context (from gateway/hermes)
        span = tracer.start_span(
            name="vllm_ascend.request",
            context=parent_ctx,
            kind=trace.SpanKind.SERVER,
        )

        # If no exporter configured, span is NonRecordingSpan — skip tracing
        # to avoid memory leak in _request_spans
        if not span.is_recording():
            return await original_method(self, request, raw_request, **kwargs)

        # Generate span_id for later retrieval (use f-string for better performance)
        span_id = f"{span.get_span_context().span_id:016x}"
        store_span(span_id, span)

        # Store span info in ContextVar (coroutine-local, no race condition)
        token = _current_span_info.set((span, span_id))
        try:
            result = await original_method(self, request, raw_request, **kwargs)
        except BaseException as e:
            pop_span(span_id)
            span.set_status(StatusCode.ERROR, str(e))
            span.end()
            raise
        finally:
            _current_span_info.reset(token)

        # (b) Validation failure: original_method returns an error response without
        # raising an exception (e.g. 400 Bad Request). The span was never passed to
        # the engine, so do_tracing will never fire — end it here to prevent a leak.
        if hasattr(result, 'status_code') and result.status_code >= 400:
            popped = pop_span(span_id)
            if popped is not None:
                span.set_status(StatusCode.ERROR, "request validation error")
                span.end()

        return result

    return wrapped


def _wrap_get_trace_headers(original_get_trace_headers):
    """Wrap _get_trace_headers to inject span context from ContextVar."""

    @functools.wraps(original_get_trace_headers)
    async def patched_get_trace_headers(self, headers):
        try:
            span_info = _current_span_info.get()
            if span_info is not None:
                span, span_id = span_info
                carrier: dict = {}
                ctx = trace.set_span_in_context(span)
                _propagator.inject(carrier, context=ctx)
                carrier["x-vllm-ascend-span-id"] = span_id
                return carrier
        except Exception as e:
            # Tracing failure should not break inference
            logger.warning(f"[tracing] Failed to generate trace headers: {e}", exc_info=True)
            # Fall through to original behavior

        # Fallback to original behavior if no span in context or on error
        return await original_get_trace_headers(self, headers)

    return patched_get_trace_headers


def patch_chat_serving(module):
    """Patch OpenAIServingChat."""
    cls = module.OpenAIServingChat
    cls.create_chat_completion = _wrap_create_method(cls.create_chat_completion)


def patch_completion_serving(module):
    """Patch OpenAIServingCompletion."""
    cls = module.OpenAIServingCompletion
    cls.create_completion = _wrap_create_method(cls.create_completion)


def patch_responses_serving(module):
    """Patch OpenAIServingResponses."""
    cls = module.OpenAIServingResponses
    cls.create_responses = _wrap_create_method(cls.create_responses)


def patch_openai_serving(module):
    """Patch the base OpenAIServing._get_trace_headers (called once)."""
    cls = module.OpenAIServing
    cls._get_trace_headers = _wrap_get_trace_headers(cls._get_trace_headers)
