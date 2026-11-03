# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.serving."""

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch as mock_patch

from vllm_ascend_tracing.patches.serving import (
    _wrap_create_method, _wrap_get_trace_headers, _current_span_info,
    patch_chat_serving, patch_completion_serving, patch_openai_serving,
)
from vllm_ascend_tracing.context import _request_spans, _lock


class TestWrapCreateMethod(unittest.TestCase):
    def setUp(self):
        with _lock:
            _request_spans.clear()

    def test_creates_span_and_stores_it(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = MagicMock(span_id=0x1234567890ABCDEF)

        original = AsyncMock(return_value="response")
        wrapped = _wrap_create_method(original)

        raw_request = MagicMock()
        raw_request.headers = {}

        with mock_patch("vllm_ascend_tracing.patches.serving.get_tracer") as mock_tracer:
            mock_tracer.return_value.start_span.return_value = mock_span
            result = asyncio.run(wrapped(MagicMock(), MagicMock(), raw_request))

        self.assertEqual(result, "response")
        original.assert_called_once()

    def test_non_recording_span_skips_tracing(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        original = AsyncMock(return_value="response")
        wrapped = _wrap_create_method(original)

        raw_request = MagicMock()
        raw_request.headers = {}

        with mock_patch("vllm_ascend_tracing.patches.serving.get_tracer") as mock_tracer:
            mock_tracer.return_value.start_span.return_value = mock_span
            result = asyncio.run(wrapped(MagicMock(), MagicMock(), raw_request))

        self.assertEqual(result, "response")

    def test_exception_ends_span_with_error(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = MagicMock(span_id=0xABCDEF)

        original = AsyncMock(side_effect=ValueError("test error"))
        wrapped = _wrap_create_method(original)

        raw_request = MagicMock()
        raw_request.headers = {}

        with mock_patch("vllm_ascend_tracing.patches.serving.get_tracer") as mock_tracer:
            mock_tracer.return_value.start_span.return_value = mock_span
            with self.assertRaises(ValueError):
                asyncio.run(wrapped(MagicMock(), MagicMock(), raw_request))

        mock_span.set_status.assert_called_once()
        mock_span.end.assert_called_once()


class TestWrapGetTraceHeaders(unittest.TestCase):
    def test_returns_carrier_with_span_info(self):
        mock_span = MagicMock()
        original = AsyncMock(return_value={})
        wrapped = _wrap_get_trace_headers(original)

        # Set contextvar
        token = _current_span_info.set((mock_span, "span-123"))
        try:
            with mock_patch("vllm_ascend_tracing.patches.serving._propagator") as mock_prop:
                result = asyncio.run(wrapped(MagicMock(), {}))
        finally:
            _current_span_info.reset(token)

        self.assertEqual(result["x-vllm-ascend-span-id"], "span-123")

    def test_falls_back_without_span_info(self):
        original = AsyncMock(return_value={"original": "headers"})
        wrapped = _wrap_get_trace_headers(original)

        result = asyncio.run(wrapped(MagicMock(), {}))
        self.assertEqual(result, {"original": "headers"})


class TestPatchFunctions(unittest.TestCase):
    def test_patch_chat_serving(self):
        module = MagicMock()
        cls = MagicMock()
        module.OpenAIServingChat = cls
        original = cls.create_chat_completion
        patch_chat_serving(module)
        self.assertIsNot(cls.create_chat_completion, original)

    def test_patch_completion_serving(self):
        module = MagicMock()
        cls = MagicMock()
        module.OpenAIServingCompletion = cls
        original = cls.create_completion
        patch_completion_serving(module)
        self.assertIsNot(cls.create_completion, original)

    def test_patch_openai_serving(self):
        module = MagicMock()
        cls = MagicMock()
        module.OpenAIServing = cls
        original = cls._get_trace_headers
        patch_openai_serving(module)
        self.assertIsNot(cls._get_trace_headers, original)


if __name__ == "__main__":
    unittest.main()
