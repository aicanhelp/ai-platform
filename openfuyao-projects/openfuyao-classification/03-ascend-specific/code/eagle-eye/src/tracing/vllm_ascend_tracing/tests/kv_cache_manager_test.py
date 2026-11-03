# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.patches.kv_cache_manager."""

import unittest
from unittest.mock import MagicMock

from vllm_ascend_tracing.patches.kv_cache_manager import (
    patch_kv_cache_manager, _empty_kv_cache_stats,
)


def _make_request(traced=True, phase="decode"):
    req = MagicMock()
    if traced:
        req.trace_headers = {"x-vllm-ascend-span-id": "test-span"}
    else:
        req.trace_headers = None
    req._tracing_phase = phase
    # Remove _vllm_ascend_stats so lazy init triggers
    if hasattr(req, '_vllm_ascend_stats'):
        del req._vllm_ascend_stats
    return req


def _make_module():
    module = MagicMock()
    kv_cls = MagicMock()
    module.KVCacheManager = kv_cls
    # Store original for reference
    kv_cls.allocate_slots = MagicMock(return_value="alloc_result")
    return module, kv_cls


class TestPatchKvCacheManager(unittest.TestCase):
    def _setup_patched(self):
        module, kv_cls = _make_module()
        patch_kv_cache_manager(module)
        # Create instance mock
        instance = MagicMock()
        instance.block_pool.get_num_free_blocks = MagicMock(side_effect=[100, 96])
        return kv_cls, instance

    def test_patch_replaces_allocate_slots(self):
        module, kv_cls = _make_module()
        original = kv_cls.allocate_slots
        patch_kv_cache_manager(module)
        self.assertIsNot(kv_cls.allocate_slots, original)

    def test_untraced_request_skips_stats(self):
        module, kv_cls = _make_module()
        original_fn = kv_cls.allocate_slots
        patch_kv_cache_manager(module)

        instance = MagicMock()
        instance.block_pool.get_num_free_blocks = MagicMock(return_value=100)
        request = _make_request(traced=False)

        # Call patched function
        result = kv_cls.allocate_slots(instance, request, 10)
        # Should not have stats
        self.assertFalse(hasattr(request, '_vllm_ascend_stats') and
                         request._vllm_ascend_stats is not None and
                         not isinstance(request._vllm_ascend_stats, MagicMock))

    def test_traced_request_accumulates_stats(self):
        module, kv_cls = _make_module()
        patch_kv_cache_manager(module)

        instance = MagicMock()
        # First call: 100 -> 96 (4 blocks allocated)
        instance.block_pool.get_num_free_blocks = MagicMock(side_effect=[100, 96])
        request = _make_request(traced=True, phase="prefill")
        # Use spec to allow hasattr to work correctly
        request.configure_mock(**{'_vllm_ascend_stats': None})
        type(request)._vllm_ascend_stats = unittest.mock.PropertyMock(side_effect=AttributeError)
        # Simpler: just use a SimpleNamespace-like approach
        request = MagicMock(spec=[])
        request.trace_headers = {"x-vllm-ascend-span-id": "test-span"}
        request._tracing_phase = "prefill"

        kv_cls.allocate_slots(instance, request, 10,
                              num_new_computed_tokens=5,
                              num_external_computed_tokens=3)

        stats = request._vllm_ascend_stats["prefill"]["kv_cache"]
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["total_blocks_allocated"], 4)
        self.assertEqual(stats["total_computed_tokens"], 5)
        self.assertEqual(stats["total_external_computed_tokens"], 3)

    def test_decode_phase_no_computed_tokens(self):
        module, kv_cls = _make_module()
        patch_kv_cache_manager(module)

        instance = MagicMock()
        instance.block_pool.get_num_free_blocks = MagicMock(side_effect=[100, 99])
        request = MagicMock(spec=[])
        request.trace_headers = {"x-vllm-ascend-span-id": "test-span"}
        request._tracing_phase = "decode"

        kv_cls.allocate_slots(instance, request, 1,
                              num_new_computed_tokens=10,
                              num_external_computed_tokens=5)

        stats = request._vllm_ascend_stats["decode"]["kv_cache"]
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["total_blocks_allocated"], 1)
        # decode phase should NOT track computed tokens
        self.assertEqual(stats["total_computed_tokens"], 0)
        self.assertEqual(stats["total_external_computed_tokens"], 0)


class TestEmptyKvCacheStats(unittest.TestCase):
    def test_structure(self):
        stats = _empty_kv_cache_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["total_blocks_allocated"], 0)
        self.assertIsInstance(stats["free_blocks_snapshots"], list)
        self.assertEqual(stats["allocation_success"], 0)


if __name__ == "__main__":
    unittest.main()
