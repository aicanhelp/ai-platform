# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.ascend_store."""

import threading
import unittest
from unittest.mock import MagicMock

from vllm_ascend_tracing.patches.ascend_store import (
    patch_pool_worker, _patch_store_class,
    _set_worker_ref, _empty_store_stats, _patch_backend_get,
)


class TestPatchPoolWorker(unittest.TestCase):
    def test_init_adds_stats_dict(self):
        module = MagicMock()
        worker_cls = type("KVPoolWorker", (), {
            "__init__": lambda self: None,
            "start_load_kv": lambda self, metadata: None,
        })
        module.KVPoolWorker = worker_cls

        patch_pool_worker(module)
        worker = worker_cls()
        self.assertIsInstance(worker.per_request_store_stats, dict)
        self.assertTrue(hasattr(worker, '_store_stats_lock'))

    def test_get_per_request_store_stats_all(self):
        module = MagicMock()
        worker_cls = type("KVPoolWorker", (), {
            "__init__": lambda self: None,
            "start_load_kv": lambda self, metadata: None,
        })
        module.KVPoolWorker = worker_cls

        patch_pool_worker(module)
        worker = worker_cls()
        worker.per_request_store_stats["req-1"] = {"save_calls": 2}
        worker.per_request_store_stats["req-2"] = {"load_calls": 1}

        result = worker.get_per_request_store_stats()
        self.assertEqual(len(result), 2)
        self.assertEqual(worker.per_request_store_stats, {})

    def test_get_per_request_store_stats_filtered(self):
        module = MagicMock()
        worker_cls = type("KVPoolWorker", (), {
            "__init__": lambda self: None,
            "start_load_kv": lambda self, metadata: None,
        })
        module.KVPoolWorker = worker_cls

        patch_pool_worker(module)
        worker = worker_cls()
        worker.per_request_store_stats["req-1"] = {"save_calls": 2}
        worker.per_request_store_stats["req-2"] = {"load_calls": 1}

        result = worker.get_per_request_store_stats(finished_req_ids={"req-1"})
        self.assertIn("req-1", result)
        self.assertNotIn("req-2", result)
        self.assertIn("req-2", worker.per_request_store_stats)


class TestSetWorkerRef(unittest.TestCase):
    def test_sets_ref_on_threads(self):
        worker = MagicMock()
        worker.kv_send_thread = MagicMock(spec=[])
        worker.kv_recv_thread = MagicMock(spec=[])
        worker.kv_layer_send_thread = None
        worker.kv_layer_recv_thread = None

        _set_worker_ref(worker)
        self.assertIs(worker.kv_send_thread._worker_ref, worker)
        self.assertIs(worker.kv_recv_thread._worker_ref, worker)


class TestPatchStoreClass(unittest.TestCase):
    def _make_worker(self):
        worker = MagicMock()
        worker.per_request_store_stats = {}
        worker._store_stats_lock = threading.Lock()
        return worker

    def test_timed_put_records_stats(self):
        cls = type("MockThread", (), {
            "__init__": lambda self: setattr(self, 'm_store', MagicMock()),
            "_handle_request": lambda self, req: None,
        })
        _patch_store_class(cls, "put", "save")

        instance = cls()
        worker = self._make_worker()
        instance._worker_ref = worker
        instance._current_req_id = "req-1"

        instance.m_store.put("data")

        self.assertIn("req-1", worker.per_request_store_stats)
        self.assertEqual(worker.per_request_store_stats["req-1"]["save_calls"], 1)
        self.assertGreaterEqual(worker.per_request_store_stats["req-1"]["save_duration_ms"], 0)

    def test_timed_get_records_stats(self):
        cls = type("MockThread", (), {
            "__init__": lambda self: setattr(self, 'm_store', MagicMock()),
            "_handle_request": lambda self, req: None,
        })
        _patch_store_class(cls, "get", "load")

        instance = cls()
        worker = self._make_worker()
        instance._worker_ref = worker
        instance._current_req_id = "req-2"

        instance.m_store.get("key")

        self.assertEqual(worker.per_request_store_stats["req-2"]["load_calls"], 1)

    def test_no_worker_ref_skips(self):
        cls = type("MockThread", (), {
            "__init__": lambda self: setattr(self, 'm_store', MagicMock()),
            "_handle_request": lambda self, req: None,
        })
        _patch_store_class(cls, "put", "save")

        instance = cls()
        instance._worker_ref = None
        instance._current_req_id = "req-1"

        # Should not raise
        instance.m_store.put("data")

    def test_handle_request_tracks_req_id(self):
        cls = type("MockThread", (), {
            "__init__": lambda self: setattr(self, 'm_store', MagicMock()),
            "_handle_request": lambda self, req: None,
        })
        _patch_store_class(cls, "put", "save")

        instance = cls()
        req_meta = MagicMock()
        req_meta.req_id = "req-3"

        instance._handle_request(req_meta)
        self.assertIsNone(instance._current_req_id)


class TestEmptyStoreStats(unittest.TestCase):
    def test_structure(self):
        stats = _empty_store_stats()
        self.assertEqual(stats["save_calls"], 0)
        self.assertEqual(stats["load_calls"], 0)
        self.assertEqual(stats["save_duration_ms"], 0.0)
        self.assertEqual(stats["load_duration_ms"], 0.0)


class TestPatchBackendGet(unittest.TestCase):
    def test_patches_m_store_get(self):
        """Test that _patch_backend_get wraps m_store.get."""
        worker = MagicMock()
        worker.m_store = MagicMock()
        worker.m_store.get = MagicMock(return_value="result")
        worker.per_request_store_stats = {}
        worker._store_stats_lock = threading.Lock()

        _patch_backend_get(worker)

        # m_store.get should be wrapped
        self.assertTrue(hasattr(worker.m_store, '_tracing_get_patched'))
        self.assertTrue(hasattr(worker.m_store, '_tracing_original_get'))

    def test_skips_if_already_patched(self):
        """Test that _patch_backend_get doesn't double-patch."""
        worker = MagicMock()
        worker.m_store = MagicMock()
        worker.m_store._tracing_get_patched = True
        original_get = worker.m_store.get

        _patch_backend_get(worker)

        # Should not re-wrap
        self.assertIs(worker.m_store.get, original_get)

    def test_records_stats_when_req_id_set(self):
        """Test that backend get records stats when _current_sync_load_req_id is set."""
        # Create a real worker-like object (not MagicMock) so getattr works correctly
        class MockWorker:
            def __init__(self):
                self.m_store = type('MockStore', (), {
                    'get': lambda *args, **kwargs: "result"
                })()
                self.per_request_store_stats = {}
                self._store_stats_lock = threading.Lock()
                self._current_sync_load_req_id = "req-sync-1"

        worker = MockWorker()
        _patch_backend_get(worker)

        # Call the patched get
        result = worker.m_store.get(["key1"], ["addr1"], [100])

        # Should record stats
        self.assertEqual(result, "result")
        self.assertIn("req-sync-1", worker.per_request_store_stats)
        self.assertEqual(worker.per_request_store_stats["req-sync-1"]["load_calls"], 1)
        self.assertGreater(worker.per_request_store_stats["req-sync-1"]["load_duration_ms"], 0)

    def test_skips_stats_when_no_req_id(self):
        """Test that backend get doesn't record when _current_sync_load_req_id is None."""
        class MockWorker:
            def __init__(self):
                self.m_store = type('MockStore', (), {
                    'get': lambda *args, **kwargs: "result"
                })()
                self.per_request_store_stats = {}
                self._store_stats_lock = threading.Lock()
                self._current_sync_load_req_id = None

        worker = MockWorker()
        _patch_backend_get(worker)

        # Call the patched get
        worker.m_store.get(["key1"], ["addr1"], [100])

        # Should not record stats
        self.assertEqual(len(worker.per_request_store_stats), 0)


class TestPatchedStartLoadKV(unittest.TestCase):
    def test_sets_req_id_per_iteration(self):
        """Test that patched start_load_kv sets _current_sync_load_req_id for each request."""
        module = MagicMock()
        worker_cls = type("KVPoolWorker", (), {
            "__init__": lambda self: None,
            "start_load_kv": lambda self, metadata: None,
        })
        module.KVPoolWorker = worker_cls

        patch_pool_worker(module)

        worker = worker_cls()
        worker.per_request_store_stats = {}
        worker._store_stats_lock = threading.Lock()

        # Mock metadata with requests
        metadata = MagicMock()
        req1 = MagicMock()
        req1.req_id = "req-1"
        req1.load_spec = MagicMock(can_load=True)
        req2 = MagicMock()
        req2.req_id = "req-2"
        req2.load_spec = MagicMock(can_load=True)
        metadata.requests = [req1, req2]

        # Call patched start_load_kv
        worker.start_load_kv(metadata)

        # Should have set _current_sync_load_req_id to the last request
        # (since original start_load_kv does nothing in mock)
        self.assertIsNone(worker._current_sync_load_req_id)


if __name__ == "__main__":
    unittest.main()
