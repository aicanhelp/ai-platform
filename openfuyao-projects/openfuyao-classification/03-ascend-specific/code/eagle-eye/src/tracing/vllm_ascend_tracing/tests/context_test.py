# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.context."""

import threading
import unittest

from vllm_ascend_tracing.context import (
    store_span, pop_span, get_tracer, _request_spans, _lock,
)


class TestSpanStorage(unittest.TestCase):
    def setUp(self):
        with _lock:
            _request_spans.clear()

    def test_store_and_pop_span(self):
        mock_span = object()
        store_span("span-1", mock_span)
        result = pop_span("span-1")
        self.assertIs(result, mock_span)

    def test_pop_nonexistent_returns_none(self):
        result = pop_span("nonexistent")
        self.assertIsNone(result)

    def test_pop_removes_span(self):
        mock_span = object()
        store_span("span-2", mock_span)
        pop_span("span-2")
        self.assertIsNone(pop_span("span-2"))

    def test_thread_safety(self):
        results = []

        def store_and_pop(span_id):
            store_span(span_id, span_id)
            results.append(pop_span(span_id))

        threads = [threading.Thread(target=store_and_pop, args=(f"s-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertIsNotNone(r)


class TestGetTracer(unittest.TestCase):
    def test_returns_tracer(self):
        tracer = get_tracer()
        self.assertIsNotNone(tracer)

    def test_tracer_has_start_span(self):
        tracer = get_tracer()
        self.assertTrue(hasattr(tracer, 'start_span'))


if __name__ == "__main__":
    unittest.main()
