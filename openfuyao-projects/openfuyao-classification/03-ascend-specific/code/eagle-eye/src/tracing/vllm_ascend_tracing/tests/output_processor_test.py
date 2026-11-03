# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.output_processor."""

import json
import unittest
from unittest.mock import MagicMock

from vllm_ascend_tracing.patches.output_processor import (
    _build_native_ctx,
    _set_phase_attrs, _create_schedule_span, _create_kv_cache_span,
    _create_store_span,
)


class TestBuildNativeCtx(unittest.TestCase):
    def test_with_stats(self):
        stats = MagicMock()
        stats.first_token_ts = 10.5
        stats.scheduled_ts = 10.0
        stats.last_token_ts = 13.5
        stats.arrival_time = 1000.0
        stats.queued_ts = 9.9
        stats.num_generation_tokens = 100

        req_state = MagicMock()
        req_state.num_cached_tokens = 50
        trace_headers = {}

        ctx = _build_native_ctx(stats, 479, trace_headers, req_state)
        self.assertEqual(ctx["num_prompt_tokens"], 479)
        self.assertEqual(ctx["num_cached_tokens"], 50)
        self.assertAlmostEqual(ctx["prefill_time_ms"], 500.0)
        self.assertAlmostEqual(ctx["decode_time_ms"], 3000.0)
        self.assertEqual(ctx["num_generation_tokens"], 100)
        self.assertIn("scheduled_ts_ns", ctx)
        self.assertIn("first_token_ts_ns", ctx)
        self.assertIn("last_token_ts_ns", ctx)

    def test_without_stats(self):
        req_state = MagicMock()
        req_state.num_cached_tokens = 0
        trace_headers = {}

        ctx = _build_native_ctx(None, 100, trace_headers, req_state)
        self.assertEqual(ctx["num_prompt_tokens"], 100)
        self.assertNotIn("prefill_time_ms", ctx)
        self.assertNotIn("scheduled_ts_ns", ctx)

    def test_with_store_stats(self):
        req_state = MagicMock()
        req_state.num_cached_tokens = 0
        store_data = {"save_calls": 1, "save_duration_ms": 3.0}
        trace_headers = {"x-ascend-store-stats": json.dumps(store_data)}

        ctx = _build_native_ctx(None, 100, trace_headers, req_state)
        self.assertEqual(ctx["store_stats"]["save_calls"], 1)

    def test_invalid_store_json(self):
        req_state = MagicMock()
        req_state.num_cached_tokens = 0
        trace_headers = {"x-ascend-store-stats": "invalid json{"}

        ctx = _build_native_ctx(None, 100, trace_headers, req_state)
        self.assertNotIn("store_stats", ctx)


class TestSetPhaseAttrs(unittest.TestCase):
    def test_prefill_attrs(self):
        span = MagicMock()
        native_ctx = {"num_prompt_tokens": 479, "prefill_time_ms": 200.0}
        _set_phase_attrs(span, "prefill", {}, native_ctx)
        span.set_attributes.assert_called_once()
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["prefill.num_tokens"], 479)
        self.assertEqual(attrs["prefill.total_time_ms"], 200.0)

    def test_decode_attrs_with_schedule(self):
        span = MagicMock()
        phase_data = {"schedule": {"total_calls": 99}}
        native_ctx = {"decode_time_ms": 3000.0}
        _set_phase_attrs(span, "decode", phase_data, native_ctx)
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["decode.num_iterations"], 99)
        self.assertAlmostEqual(attrs["decode.avg_time_per_token_ms"], 3000.0 / 99)

    def test_decode_attrs_no_schedule(self):
        span = MagicMock()
        phase_data = {"schedule": None}
        native_ctx = {"decode_time_ms": 1000.0}
        _set_phase_attrs(span, "decode", phase_data, native_ctx)
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["decode.num_iterations"], 0)


class TestCreateScheduleSpan(unittest.TestCase):
    def test_creates_span_with_attrs(self):
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        phase_ctx = MagicMock()
        data = {
            "total_calls": 5, "total_tokens": 50,
            "avg_tokens_per_call": 10.0, "total_time_ms": 8.0,
            "avg_time_ms": 1.6, "max_time_ms": 3.0,
            "avg_queue_length": 2.0, "max_queue_length": 4,
            "avg_running_count": 1.5,
        }
        _create_schedule_span(tracer, phase_ctx, "decode", data, 1000)
        tracer.start_span.assert_called_once()
        span.set_attributes.assert_called_once()
        span.end.assert_called_once_with(end_time=1000)


class TestCreateKvCacheSpan(unittest.TestCase):
    def test_prefill_with_cache_hit(self):
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        phase_ctx = MagicMock()
        kv_data = {
            "total_calls": 1, "total_blocks_allocated": 4,
            "final_free_blocks": 83, "min_free_blocks": 83,
            "avg_free_blocks": 83.0,
            "total_external_computed_tokens": 479,
        }
        native_ctx = {"num_prompt_tokens": 479, "num_cached_tokens": 384}
        _create_kv_cache_span(tracer, phase_ctx, "prefill", kv_data, native_ctx, 1000, 0)
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["kv_cache.total_cached_tokens"], 384)
        self.assertAlmostEqual(attrs["kv_cache.cache_hit_rate"], 384 / 479)
        self.assertTrue(attrs["kv_cache.has_external_cache"])
        span.end.assert_called_once_with(end_time=1000)

    def test_decode_with_preemptions(self):
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        phase_ctx = MagicMock()
        kv_data = {
            "total_calls": 99, "total_blocks_allocated": 1,
            "final_free_blocks": 82, "min_free_blocks": 82,
            "avg_free_blocks": 82.5,
        }
        native_ctx = {"num_prompt_tokens": 479, "num_cached_tokens": 0}
        _create_kv_cache_span(tracer, phase_ctx, "decode", kv_data, native_ctx, 1000, 3)
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["kv_cache.total_preemptions"], 3)
        self.assertAlmostEqual(attrs["kv_cache.preemption_rate"], 3 / 99)


class TestCreateStoreSpan(unittest.TestCase):
    def test_no_store_stats(self):
        tracer = MagicMock()
        native_ctx = {}
        _create_store_span(tracer, MagicMock(), native_ctx, 1000)
        tracer.start_span.assert_not_called()

    def test_with_save_stats(self):
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        native_ctx = {
            "store_stats": {
                "save_calls": 2, "save_duration_ms": 6.0,
                "save_max_duration_ms": 4.0,
                "load_calls": 0, "load_duration_ms": 0.0,
                "load_max_duration_ms": 0.0,
            }
        }
        _create_store_span(tracer, MagicMock(), native_ctx, 5000)
        tracer.start_span.assert_called_once()
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["ascend_store.save_calls"], 2)
        self.assertAlmostEqual(attrs["ascend_store.save_avg_duration_ms"], 3.0)
        span.end.assert_called_once_with(end_time=5000)

    def test_zero_duration_not_shown(self):
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        native_ctx = {
            "store_stats": {
                "save_calls": 1, "save_duration_ms": 0.0,
                "save_max_duration_ms": 0.0,
                "load_calls": 0, "load_duration_ms": 0.0,
                "load_max_duration_ms": 0.0,
            }
        }
        _create_store_span(tracer, MagicMock(), native_ctx, 5000)
        attrs = span.set_attributes.call_args[0][0]
        self.assertEqual(attrs["ascend_store.save_calls"], 1)
        self.assertNotIn("ascend_store.save_duration_ms", attrs)


if __name__ == "__main__":
    unittest.main()
