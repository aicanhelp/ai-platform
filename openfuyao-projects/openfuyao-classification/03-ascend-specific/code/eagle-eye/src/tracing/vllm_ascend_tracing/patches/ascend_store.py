# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patches for AscendStore Worker-level tracing.

Records put/get operation durations per request into worker.per_request_store_stats.
Stats are picked up by _get_kv_connector_output (kv_connector.py) and flow
through ModelRunnerOutput → Aggregator → Scheduler.
"""

import time
import threading
import functools
import logging

logger = logging.getLogger("vllm_ascend.tracing")


def patch_pool_worker(module):
    """Patch KVPoolWorker to initialize per-request stats collection."""
    worker_cls = module.KVPoolWorker
    original_init = worker_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.per_request_store_stats: dict = {}
        self._store_stats_lock = threading.Lock()
        _set_worker_ref(self)
        _patch_backend_get(self)

    worker_cls.__init__ = patched_init

    # Also patch register_kv_caches since threads might be created there
    if hasattr(worker_cls, 'register_kv_caches'):
        original_register = worker_cls.register_kv_caches

        @functools.wraps(original_register)
        def patched_register(self, *args, **kwargs):
            result = original_register(self, *args, **kwargs)
            _set_worker_ref(self)
            _patch_backend_get(self)
            return result

        worker_cls.register_kv_caches = patched_register

    def get_per_request_store_stats(self, finished_req_ids=None):
        """Get and clear stats. If finished_req_ids given, only pop those."""
        with self._store_stats_lock:
            if finished_req_ids is None:
                result = dict(self.per_request_store_stats)
                self.per_request_store_stats.clear()
                return result
            result = {}
            for req_id in finished_req_ids:
                if req_id in self.per_request_store_stats:
                    result[req_id] = self.per_request_store_stats.pop(req_id)
            return result

    worker_cls.get_per_request_store_stats = get_per_request_store_stats

    # Patch start_load_kv to set _current_sync_load_req_id per request iteration.
    # In sync mode (load_async=False), m_store.get is called directly inside
    # start_load_kv's loop. We wrap it to track which req_id is active.
    original_start_load = worker_cls.start_load_kv

    @functools.wraps(original_start_load)
    def patched_start_load_kv(self, metadata):
        # Wrap metadata.requests so each iteration sets the active req_id
        original_requests = metadata.requests
        worker_self = self

        class _TracingRequestIter:
            """Iterator wrapper that sets _current_sync_load_req_id per request."""
            def __init__(self, requests):
                self._requests = requests
                self._iter = iter(requests)

            def __iter__(self):
                return self

            def __next__(self):
                request = next(self._iter)
                worker_self._current_sync_load_req_id = request.req_id
                return request

            def __len__(self):
                return len(self._requests)

        metadata.requests = _TracingRequestIter(original_requests)
        try:
            original_start_load(self, metadata)
        finally:
            self._current_sync_load_req_id = None
            metadata.requests = original_requests

    worker_cls.start_load_kv = patched_start_load_kv


def _patch_backend_get(worker):
    """Patch m_store.get directly on the worker so sync load path is captured.

    This covers the sync load path in start_load_kv() which bypasses
    KVCacheStoreRecvingThread when load_async=False (default).
    Only records stats when _current_sync_load_req_id is set (sync path),
    so it won't double-count with the async thread patch.
    """
    m_store = getattr(worker, 'm_store', None)
    if m_store is None or getattr(m_store, '_tracing_get_patched', False):
        return

    original_get = m_store.get

    def timed_get(*args, **kwargs):
        # Only record when in sync load path (req_id set by patched_start_load_kv)
        req_id = getattr(worker, '_current_sync_load_req_id', None)
        if req_id:
            with worker._store_stats_lock:
                if req_id not in worker.per_request_store_stats:
                    worker.per_request_store_stats[req_id] = _empty_store_stats()
                worker.per_request_store_stats[req_id]["load_calls"] += 1

        start = time.perf_counter()
        result = original_get(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        if req_id:
            with worker._store_stats_lock:
                if req_id in worker.per_request_store_stats:
                    stats = worker.per_request_store_stats[req_id]
                    stats["load_duration_ms"] += duration_ms
                    stats["load_max_duration_ms"] = max(
                        stats["load_max_duration_ms"], duration_ms
                    )
        return result

    m_store.get = timed_get
    m_store._tracing_get_patched = True
    m_store._tracing_original_get = original_get


def _set_worker_ref(worker):
    """Set back-reference on all thread attributes so they can write stats."""
    for attr_name in ('kv_send_thread', 'kv_recv_thread'):
        thread = getattr(worker, attr_name, None)
        if thread is not None and not hasattr(thread, '_worker_ref'):
            thread._worker_ref = worker


def patch_kv_transfer_threads(module):
    """Patch Backend.put/get on transfer threads to record only actual IO."""
    _patch_sending_thread(module)
    _patch_receiving_thread(module)


def _patch_sending_thread(module):
    _patch_store_class(
        module.KVCacheStoreSendingThread, "put", "save",
    )
    layer_cls = getattr(module, 'KVCacheStoreLayerSendingThread', None)
    if layer_cls is not None:
        _patch_store_class(layer_cls, "put", "save")


def _patch_receiving_thread(module):
    _patch_store_class(
        module.KVCacheStoreRecvingThread, "get", "load",
    )
    layer_cls = getattr(module, 'KVCacheStoreLayerRecvingThread', None)
    if layer_cls is not None:
        _patch_store_class(layer_cls, "get", "load")


def _patch_store_class(cls, method_name, op_type):
    """Generic patcher for sending/receiving thread classes.

    Wraps m_store.put/get to record timing to worker.per_request_store_stats,
    and wraps _handle_request to track current request ID.
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Use the true original method, not the backend_get wrapper
        tracing_original = getattr(self.m_store, f'_tracing_original_{method_name}', None)
        original_method = tracing_original if tracing_original else getattr(self.m_store, method_name)

        def timed_method(*m_args, **m_kwargs):
            req_id = getattr(self, '_current_req_id', None)
            worker = getattr(self, '_worker_ref', None)

            # Increment count BEFORE operation (guaranteed capture)
            if req_id and worker:
                with worker._store_stats_lock:
                    if req_id not in worker.per_request_store_stats:
                        worker.per_request_store_stats[req_id] = _empty_store_stats()
                    worker.per_request_store_stats[req_id][f"{op_type}_calls"] += 1
            elif req_id and not worker:
                logger.warning("[tracing] %s called but _worker_ref not set on thread", method_name)

            start = time.perf_counter()
            result = original_method(*m_args, **m_kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            # Add duration AFTER operation (best-effort)
            if req_id and worker:
                with worker._store_stats_lock:
                    if req_id in worker.per_request_store_stats:
                        stats = worker.per_request_store_stats[req_id]
                        stats[f"{op_type}_duration_ms"] += duration_ms
                        stats[f"{op_type}_max_duration_ms"] = max(
                            stats[f"{op_type}_max_duration_ms"], duration_ms
                        )
            return result

        setattr(self.m_store, method_name, timed_method)

    cls.__init__ = patched_init

    original_handle = cls._handle_request

    @functools.wraps(original_handle)
    def patched_handle(self, req_meta):
        self._current_req_id = req_meta.req_id
        try:
            original_handle(self, req_meta)
        finally:
            self._current_req_id = None

    cls._handle_request = patched_handle


def _empty_store_stats():
    return {
        "save_calls": 0, "save_duration_ms": 0.0, "save_max_duration_ms": 0.0,
        "load_calls": 0, "load_duration_ms": 0.0, "load_max_duration_ms": 0.0,
    }
