# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.kv_connector."""

import unittest
from unittest.mock import MagicMock, patch as mock_patch
from contextlib import contextmanager

from vllm_ascend_tracing.patches.kv_connector import (
    patch_kv_connector_mixin, patch_kv_output_aggregator,
    _find_worker_with_stats, _merge_tp_stats,
)


class TestFindWorkerWithStats(unittest.TestCase):
    def test_direct_connector_worker(self):
        connector = MagicMock()
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_per_request_store_stats = MagicMock()
        result = _find_worker_with_stats(connector)
        self.assertIs(result, connector.connector_worker)

    def test_multi_connector(self):
        connector = MagicMock(spec=[])
        connector._connectors = [MagicMock(spec=[]), MagicMock()]
        connector._connectors[0].connector_worker = None
        worker = MagicMock()
        worker.get_per_request_store_stats = MagicMock()
        connector._connectors[1].connector_worker = worker
        result = _find_worker_with_stats(connector)
        self.assertIs(result, worker)

    def test_no_worker_found(self):
        connector = MagicMock(spec=[])
        result = _find_worker_with_stats(connector)
        self.assertIsNone(result)


class TestMergeTpStats(unittest.TestCase):
    def test_takes_max(self):
        existing = {
            "save_calls": 2, "save_duration_ms": 5.0, "save_max_duration_ms": 4.0,
            "load_calls": 1, "load_duration_ms": 3.0, "load_max_duration_ms": 3.0,
        }
        new = {
            "save_calls": 3, "save_duration_ms": 8.0, "save_max_duration_ms": 6.0,
            "load_calls": 1, "load_duration_ms": 2.0, "load_max_duration_ms": 2.0,
        }
        _merge_tp_stats(existing, new)
        self.assertEqual(existing["save_calls"], 3)  # max(2, 3)
        self.assertEqual(existing["save_duration_ms"], 8.0)  # max(5, 8)
        self.assertEqual(existing["save_max_duration_ms"], 6.0)  # max(4, 6)
        self.assertEqual(existing["load_calls"], 1)  # max(1, 1)
        self.assertEqual(existing["load_duration_ms"], 3.0)  # max(3, 2)


class TestPatchKvOutputAggregator(unittest.TestCase):
    def test_aggregates_store_stats(self):
        module = MagicMock()
        aggregator_cls = MagicMock()
        aggregator_cls.__name__ = "KVOutputAggregator"
        module.KVOutputAggregator = aggregator_cls

        # Mock from_connector
        aggregator_cls.from_connector = MagicMock()

        # Original aggregate returns a result with kv_connector_output
        result = MagicMock()
        result.kv_connector_output = MagicMock(spec=[])
        aggregator_cls.aggregate = MagicMock(return_value=result)

        patch_kv_output_aggregator(module)

        # Create mock outputs from 2 TP ranks
        output1 = MagicMock()
        output1.kv_connector_output = MagicMock()
        output1.kv_connector_output.per_request_store_stats = {
            "req-1": {"save_calls": 1, "save_duration_ms": 5.0, "save_max_duration_ms": 5.0,
                      "load_calls": 0, "load_duration_ms": 0.0, "load_max_duration_ms": 0.0}
        }
        output2 = MagicMock()
        output2.kv_connector_output = MagicMock()
        output2.kv_connector_output.per_request_store_stats = {
            "req-1": {"save_calls": 1, "save_duration_ms": 8.0, "save_max_duration_ms": 8.0,
                      "load_calls": 0, "load_duration_ms": 0.0, "load_max_duration_ms": 0.0}
        }

        # Call patched aggregate
        instance = MagicMock()
        aggregator_cls.aggregate(instance, [output1, output2], output_rank=0)

    def test_no_store_stats(self):
        module = MagicMock()
        aggregator_cls = MagicMock()
        aggregator_cls.__name__ = "KVOutputAggregator"
        module.KVOutputAggregator = aggregator_cls

        # Mock from_connector
        aggregator_cls.from_connector = MagicMock()

        result = MagicMock()
        result.kv_connector_output = None
        aggregator_cls.aggregate = MagicMock(return_value=result)

        patch_kv_output_aggregator(module)
        # Should not raise
        instance = MagicMock()
        aggregator_cls.aggregate(instance, [MagicMock()], output_rank=0)


class TestPatchKvConnectorMixin(unittest.TestCase):
    def test_patches_method(self):
        module = MagicMock()
        mixin_cls = MagicMock()
        module.KVConnectorModelRunnerMixin = mixin_cls

        @staticmethod
        @contextmanager
        def original_get_output(scheduler_output, wait_for_save=True, defer_finalize=False):
            output = MagicMock(spec=[])
            yield output

        mixin_cls._get_kv_connector_output = original_get_output

        patch_kv_connector_mixin(module)
        self.assertIsNot(mixin_cls._get_kv_connector_output, original_get_output)

    def test_collects_stats_when_available(self):
        """Test that stats are collected when worker has them."""
        module = MagicMock()
        mixin_cls = type("KVConnectorModelRunnerMixin", (), {})
        module.KVConnectorModelRunnerMixin = mixin_cls

        # Create a real context manager that we can patch
        original_called = []

        @staticmethod
        @contextmanager
        def mock_original(scheduler_output, wait_for_save=True, defer_finalize=False):
            output = MagicMock()
            output.per_request_store_stats = None
            original_called.append((wait_for_save, defer_finalize))
            yield output

        mixin_cls._get_kv_connector_output = mock_original
        patch_kv_connector_mixin(module)

        # Mock vllm.distributed.kv_transfer.get_kv_transfer_group (imported inside patched function)
        with mock_patch('vllm.distributed.kv_transfer.get_kv_transfer_group') as mock_get_group:
            mock_connector = MagicMock()
            mock_worker = MagicMock()
            mock_worker.get_per_request_store_stats.return_value = {"req-1": {"save_calls": 1}}
            mock_connector.connector_worker = mock_worker
            mock_get_group.return_value = mock_connector

            patched_method = mixin_cls._get_kv_connector_output
            with patched_method(MagicMock(), wait_for_save=True, defer_finalize=False) as output:
                pass

            # Stats should be attached
            self.assertEqual(output.per_request_store_stats, {"req-1": {"save_calls": 1}})

    def test_skips_empty_stats(self):
        """Test that empty stats are not attached."""
        module = MagicMock()
        mixin_cls = type("KVConnectorModelRunnerMixin", (), {})
        module.KVConnectorModelRunnerMixin = mixin_cls

        @staticmethod
        @contextmanager
        def mock_original(scheduler_output, wait_for_save=True, defer_finalize=False):
            output = MagicMock()
            output.per_request_store_stats = None
            yield output

        mixin_cls._get_kv_connector_output = mock_original
        patch_kv_connector_mixin(module)

        with mock_patch('vllm.distributed.kv_transfer.get_kv_transfer_group') as mock_get_group:
            mock_connector = MagicMock()
            mock_worker = MagicMock()
            mock_worker.get_per_request_store_stats.return_value = {}  # Empty
            mock_connector.connector_worker = mock_worker
            mock_get_group.return_value = mock_connector

            patched_method = mixin_cls._get_kv_connector_output
            with patched_method(MagicMock(), wait_for_save=True, defer_finalize=False) as output:
                pass

            # Stats should remain None
            self.assertIsNone(output.per_request_store_stats)


if __name__ == "__main__":
    unittest.main()
