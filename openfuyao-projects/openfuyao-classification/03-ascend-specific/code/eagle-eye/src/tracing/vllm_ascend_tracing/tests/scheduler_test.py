# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.scheduler."""

import unittest
from unittest.mock import MagicMock

from vllm_ascend_tracing.patches.scheduler import (
    patch_scheduler, _empty_schedule_stats, _finalize_stats, _merge_store_stats,
)


def _make_request(num_computed=0, num_prompt=100, traced=True):
    req = MagicMock()
    req.num_computed_tokens = num_computed
    req.num_prompt_tokens = num_prompt
    req.num_preemptions = 0
    if traced:
        req.trace_headers = {"x-vllm-ascend-span-id": "span-1"}
    else:
        req.trace_headers = None
    # Ensure _vllm_ascend_stats doesn't exist as a mock
    if hasattr(req, '_vllm_ascend_stats'):
        del req._vllm_ascend_stats
    return req


class TestEmptyScheduleStats(unittest.TestCase):
    def test_structure(self):
        stats = _empty_schedule_stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["total_time_ms"], 0.0)
        self.assertEqual(stats["max_time_ms"], 0.0)
        self.assertEqual(stats["total_tokens"], 0)
        self.assertIsInstance(stats["queue_lengths"], list)
        self.assertIsInstance(stats["running_counts"], list)


class TestFinalizeStats(unittest.TestCase):
    def test_empty_stats(self):
        stats = {
            "num_preemptions": 0,
            "prefill": {"schedule": _empty_schedule_stats(), "kv_cache": None},
            "decode": {"schedule": _empty_schedule_stats(), "kv_cache": None},
        }
        result = _finalize_stats(stats)
        self.assertEqual(result["num_preemptions"], 0)
        self.assertIsNone(result["prefill"]["schedule"])
        self.assertIsNone(result["decode"]["schedule"])

    def test_with_schedule_data(self):
        sched = _empty_schedule_stats()
        sched["count"] = 5
        sched["total_time_ms"] = 10.0
        sched["max_time_ms"] = 3.0
        sched["total_tokens"] = 50
        sched["queue_lengths"] = [1, 2, 3, 2, 1]
        sched["running_counts"] = [1, 1, 2, 2, 2]

        stats = {
            "num_preemptions": 2,
            "prefill": {"schedule": sched, "kv_cache": None},
            "decode": {"schedule": _empty_schedule_stats(), "kv_cache": None},
        }
        result = _finalize_stats(stats)
        self.assertEqual(result["num_preemptions"], 2)

        ps = result["prefill"]["schedule"]
        self.assertEqual(ps["total_calls"], 5)
        self.assertEqual(ps["total_tokens"], 50)
        self.assertAlmostEqual(ps["avg_tokens_per_call"], 10.0)
        self.assertAlmostEqual(ps["avg_time_ms"], 2.0)
        self.assertAlmostEqual(ps["max_time_ms"], 3.0)
        self.assertAlmostEqual(ps["avg_queue_length"], 1.8)
        self.assertEqual(ps["max_queue_length"], 3)
        self.assertAlmostEqual(ps["avg_running_count"], 1.6)

    def test_with_kv_cache_data(self):
        kv = {
            "total_calls": 3,
            "total_blocks_allocated": 12,
            "free_blocks_snapshots": [100, 96, 90],
            "allocation_success": 3,
            "total_computed_tokens": 50,
            "total_external_computed_tokens": 10,
        }
        stats = {
            "num_preemptions": 0,
            "prefill": {"schedule": None, "kv_cache": kv},
            "decode": {"schedule": None, "kv_cache": None},
        }
        result = _finalize_stats(stats)
        pk = result["prefill"]["kv_cache"]
        self.assertEqual(pk["total_calls"], 3)
        self.assertEqual(pk["total_blocks_allocated"], 12)
        self.assertAlmostEqual(pk["avg_blocks_per_call"], 4.0)
        self.assertEqual(pk["final_free_blocks"], 90)
        self.assertEqual(pk["min_free_blocks"], 90)
        self.assertAlmostEqual(pk["avg_free_blocks"], 95.333, places=2)


class TestMergeStoreStats(unittest.TestCase):
    def test_basic_merge(self):
        existing = {
            "save_calls": 1, "save_duration_ms": 5.0, "save_max_duration_ms": 5.0,
            "load_calls": 0, "load_duration_ms": 0.0, "load_max_duration_ms": 0.0,
        }
        new = {
            "save_calls": 2, "save_duration_ms": 8.0, "save_max_duration_ms": 6.0,
            "load_calls": 1, "load_duration_ms": 3.0, "load_max_duration_ms": 3.0,
        }
        _merge_store_stats(existing, new)
        self.assertEqual(existing["save_calls"], 3)
        self.assertAlmostEqual(existing["save_duration_ms"], 13.0)
        self.assertAlmostEqual(existing["save_max_duration_ms"], 6.0)
        self.assertEqual(existing["load_calls"], 1)
        self.assertAlmostEqual(existing["load_duration_ms"], 3.0)


class TestPatchSchedule(unittest.TestCase):
    def _make_scheduler_module(self):
        # Create a real class instead of MagicMock to test patching
        class MockScheduler:
            def __init__(self):
                self.waiting = []
                self.running = []
                self.requests = {}

            def schedule(self):
                output = MagicMock()
                output.num_scheduled_tokens = {"req-1": 100}
                return output

            def update_from_output(self, scheduler_output, model_runner_output):
                pass

        module = MagicMock()
        module.Scheduler = MockScheduler
        return module, MockScheduler

    def test_patch_replaces_methods(self):
        module, cls = self._make_scheduler_module()
        original_schedule = cls.schedule
        original_update = cls.update_from_output
        patch_scheduler(module)
        self.assertIsNot(cls.schedule, original_schedule)
        self.assertIsNot(cls.update_from_output, original_update)


if __name__ == "__main__":
    unittest.main()
