# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patch for KVCacheManager.allocate_slots().

Accumulates KV cache allocation statistics per request,
separated by prefill/decode phase.
"""

import functools
import inspect


def patch_kv_cache_manager(module):
    """Patch the KVCacheManager class after module is loaded."""
    kv_cache_cls = module.KVCacheManager
    original_allocate = kv_cache_cls.allocate_slots

    @functools.wraps(original_allocate)
    def patched_allocate_slots(self, request, num_new_tokens,
                               num_new_computed_tokens=0,
                               new_computed_blocks=None,
                               num_lookahead_tokens=0,
                               num_external_computed_tokens=0,
                               delay_cache_blocks=False,
                               num_encoder_tokens=0,
                               full_sequence_must_fit=False,
                               **extra_kwargs):
        free_blocks_before = self.block_pool.get_num_free_blocks()

        all_kwargs = {
            'num_new_computed_tokens': num_new_computed_tokens,
            'new_computed_blocks': new_computed_blocks,
            'num_lookahead_tokens': num_lookahead_tokens,
            'num_external_computed_tokens': num_external_computed_tokens,
            'delay_cache_blocks': delay_cache_blocks,
            'num_encoder_tokens': num_encoder_tokens,
            'full_sequence_must_fit': full_sequence_must_fit,
        }
        all_kwargs.update(extra_kwargs)

        sig = inspect.signature(original_allocate)
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in sig.parameters
        }

        result = original_allocate(self, request, num_new_tokens, **filtered_kwargs)

        # Skip if tracing not enabled
        if not (request.trace_headers
                and "x-vllm-ascend-span-id" in request.trace_headers):
            return result

        blocks_allocated = free_blocks_before - self.block_pool.get_num_free_blocks()

        # Lazy init stats on request
        if not hasattr(request, '_vllm_ascend_stats'):
            request._vllm_ascend_stats = {
                "prefill": {"schedule": None, "kv_cache": _empty_kv_cache_stats()},
                "decode": {"schedule": None, "kv_cache": _empty_kv_cache_stats()},
            }

        phase = getattr(request, '_tracing_phase', None) or "decode"

        if request._vllm_ascend_stats[phase].get("kv_cache") is None:
            request._vllm_ascend_stats[phase]["kv_cache"] = _empty_kv_cache_stats()

        kv = request._vllm_ascend_stats[phase]["kv_cache"]
        kv["total_calls"] += 1
        kv["total_blocks_allocated"] += blocks_allocated
        kv["free_blocks_snapshots"].append(
            free_blocks_before - blocks_allocated
        )
        kv["allocation_success"] += (1 if result is not None else 0)

        if phase == "prefill":
            kv["total_computed_tokens"] += num_new_computed_tokens
            kv["total_external_computed_tokens"] += num_external_computed_tokens

        return result

    kv_cache_cls.allocate_slots = patched_allocate_slots


def _empty_kv_cache_stats():
    return {
        "total_calls": 0,
        "total_blocks_allocated": 0,
        "free_blocks_snapshots": [],
        "allocation_success": 0,
        "total_computed_tokens": 0,
        "total_external_computed_tokens": 0,
    }
