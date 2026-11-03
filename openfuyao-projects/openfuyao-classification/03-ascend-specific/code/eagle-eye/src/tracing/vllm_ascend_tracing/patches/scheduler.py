# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patch for Scheduler.schedule() and Scheduler.update_from_output().

schedule() — Accumulates per-request scheduling statistics separated by
prefill/decode phase. Phase is determined by snapshotting num_computed_tokens
BEFORE schedule() runs.

update_from_output() — Injects accumulated stats as JSON into
request.trace_headers before the original method reads them.
"""

import json
import sys
import time
import functools
import logging

logger = logging.getLogger("vllm_ascend.tracing")

# Store original unpatched methods to enable unpatching
_original_methods = {}


def patch_scheduler(module):
    scheduler_cls = module.Scheduler
    _patch_schedule(scheduler_cls)
    _patch_update_from_output(scheduler_cls)


def patch_balance_scheduler(module):
    """Patch BalanceScheduler class for vLLM-Ascend 0.23.0+ compatibility."""
    try:
        balance_scheduler_cls = module.BalanceScheduler

        # Check if Scheduler was already patched
        # If so, unpatch it to avoid double instrumentation
        for parent_cls in balance_scheduler_cls.__mro__[1:]:  # Skip BalanceScheduler itself
            if parent_cls.__name__ == 'Scheduler' and hasattr(parent_cls.schedule, '_vllm_ascend_tracing_patched'):
                logger.info(f"[tracing] Unpatching parent Scheduler to avoid double wrapping")
                # Restore original method
                if id(parent_cls) in _original_methods:
                    parent_cls.schedule = _original_methods[id(parent_cls)]['schedule']

        _patch_schedule(balance_scheduler_cls)
        _patch_update_from_output(balance_scheduler_cls)
    except AttributeError:
        # BalanceScheduler may not exist in older versions
        logger.warning(f"[tracing] BalanceScheduler not found in {module.__name__}")
        pass
    except Exception as e:
        logger.warning(f"[tracing] Error patching BalanceScheduler: {e}")



def _patch_schedule(scheduler_cls):
    original_schedule = scheduler_cls.schedule

    # Check if already patched by checking for our marker attribute
    if hasattr(original_schedule, '_vllm_ascend_tracing_patched'):
        return

    # Save original method for potential unpatching
    if id(scheduler_cls) not in _original_methods:
        _original_methods[id(scheduler_cls)] = {'schedule': original_schedule}

    @functools.wraps(original_schedule)
    def patched_schedule(self):
        start_time = time.perf_counter()
        queue_length = len(self.waiting)
        running_count = len(self.running)

        # Snapshot phase BEFORE schedule() updates num_computed_tokens
        prefill_snapshot = {}
        for req_id, request in self.requests.items():
            if (request.trace_headers
                    and "x-vllm-ascend-span-id" in request.trace_headers):
                is_prefill = request.num_computed_tokens < request.num_prompt_tokens
                prefill_snapshot[req_id] = is_prefill
                request._tracing_phase = "prefill" if is_prefill else "decode"

        output = original_schedule(self)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        for req_id, num_tokens in output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            if not (request.trace_headers
                    and "x-vllm-ascend-span-id" in request.trace_headers):
                continue

            # Lazy init
            if not hasattr(request, '_vllm_ascend_stats'):
                request._vllm_ascend_stats = {
                    "prefill": {"schedule": _empty_schedule_stats(), "kv_cache": None},
                    "decode": {"schedule": _empty_schedule_stats(), "kv_cache": None},
                }
            for p in ("prefill", "decode"):
                if request._vllm_ascend_stats.get(p, {}).get("schedule") is None:
                    request._vllm_ascend_stats.setdefault(p, {})["schedule"] = _empty_schedule_stats()

            phase = "prefill" if prefill_snapshot.get(req_id, False) else "decode"
            s = request._vllm_ascend_stats[phase]["schedule"]
            s["count"] += 1
            s["total_time_ms"] += elapsed_ms
            s["max_time_ms"] = max(s["max_time_ms"], elapsed_ms)
            s["total_tokens"] += num_tokens
            s["queue_lengths"].append(queue_length)
            s["running_counts"].append(running_count)

        return output

    patched_schedule._vllm_ascend_tracing_patched = True
    scheduler_cls.schedule = patched_schedule


def _patch_update_from_output(scheduler_cls):
    original_update = scheduler_cls.update_from_output
    _pending_store_stats = {}

    @functools.wraps(original_update)
    def patched_update_from_output(self, scheduler_output, model_runner_output):
        try:
            # Inject accumulated tracing stats into trace_headers
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is None or not hasattr(request, '_vllm_ascend_stats'):
                    continue
                if not request.trace_headers:
                    continue
                if not isinstance(request.trace_headers, dict):
                    request.trace_headers = dict(request.trace_headers)

                stats = request._vllm_ascend_stats
                stats["num_preemptions"] = request.num_preemptions
                request.trace_headers["x-vllm-ascend-stats"] = json.dumps(
                    _finalize_stats(stats)
                )

            # Collect AscendStore stats from KVConnectorOutput pipeline
            store_stats = None
            kv_connector_output = getattr(model_runner_output, 'kv_connector_output', None)
            if kv_connector_output is not None:
                # Read from the official field (now serializable via outputs.py patch)
                store_stats = getattr(kv_connector_output, 'per_request_store_stats', None)
                if store_stats:
                    logger.debug("[tracing] scheduler: received store_stats for %d requests: %s",
                                 len(store_stats), list(store_stats.keys()))

            # Merge into pending buffer
            if store_stats:
                for req_id, new_stats in store_stats.items():
                    if req_id in _pending_store_stats:
                        _merge_store_stats(_pending_store_stats[req_id], new_stats)
                    else:
                        _pending_store_stats[req_id] = dict(new_stats)

            # Write buffered stats to trace_headers for alive requests
            for req_id in list(_pending_store_stats.keys()):
                request = self.requests.get(req_id)
                if request is None:
                    logger.debug("[tracing] scheduler: req_id=%s not in self.requests (request ended)", req_id)
                    _pending_store_stats.pop(req_id, None)
                    continue
                if not request.trace_headers:
                    logger.debug("[tracing] scheduler: req_id=%s has no trace_headers", req_id)
                    _pending_store_stats.pop(req_id, None)
                    continue
                if "x-vllm-ascend-span-id" not in request.trace_headers:
                    logger.debug("[tracing] scheduler: req_id=%s missing x-vllm-ascend-span-id", req_id)
                    _pending_store_stats.pop(req_id, None)
                    continue
                if not isinstance(request.trace_headers, dict):
                    request.trace_headers = dict(request.trace_headers)
                request.trace_headers["x-ascend-store-stats"] = json.dumps(
                    _pending_store_stats[req_id]
                )
                logger.debug("[tracing] scheduler: wrote x-ascend-store-stats for req_id=%s", req_id)
        except Exception:
            logger.warning_once(
                "[tracing] patched_update_from_output: unexpected error in tracing hook "
                "(suppressed to avoid disrupting inference); check for vllm version drift",
                exc_info=True,
            )

        return original_update(self, scheduler_output, model_runner_output)

    scheduler_cls.update_from_output = patched_update_from_output


def _merge_store_stats(existing, new_stats):
    """Merge new AscendStore stats into existing (accumulate)."""
    existing["save_calls"] += new_stats.get("save_calls", 0)
    existing["save_duration_ms"] += new_stats.get("save_duration_ms", 0.0)
    existing["save_max_duration_ms"] = max(
        existing["save_max_duration_ms"], new_stats.get("save_max_duration_ms", 0.0)
    )
    existing["load_calls"] += new_stats.get("load_calls", 0)
    existing["load_duration_ms"] += new_stats.get("load_duration_ms", 0.0)
    existing["load_max_duration_ms"] = max(
        existing["load_max_duration_ms"], new_stats.get("load_max_duration_ms", 0.0)
    )


def _empty_schedule_stats():
    return {
        "count": 0,
        "total_time_ms": 0.0,
        "max_time_ms": 0.0,
        "total_tokens": 0,
        "queue_lengths": [],
        "running_counts": [],
    }


def _finalize_stats(stats):
    """Convert raw accumulated stats into final aggregated form."""
    result = {"num_preemptions": stats.get("num_preemptions", 0)}

    for phase in ("prefill", "decode"):
        phase_data = stats.get(phase, {})
        result[phase] = {}

        # Schedule
        sched = phase_data.get("schedule") or {}
        count = sched.get("count", 0)
        if count > 0:
            ql = sched.get("queue_lengths", [])
            rc = sched.get("running_counts", [])
            result[phase]["schedule"] = {
                "total_calls": count,
                "total_tokens": sched["total_tokens"],
                "avg_tokens_per_call": sched["total_tokens"] / count,
                "total_time_ms": sched["total_time_ms"],
                "avg_time_ms": sched["total_time_ms"] / count,
                "max_time_ms": sched["max_time_ms"],
                "avg_queue_length": sum(ql) / len(ql) if ql else 0,
                "max_queue_length": max(ql) if ql else 0,
                "avg_running_count": sum(rc) / len(rc) if rc else 0,
            }
        else:
            result[phase]["schedule"] = None

        # KV cache
        kv = phase_data.get("kv_cache") or {}
        if kv.get("total_calls", 0) > 0:
            snaps = kv.get("free_blocks_snapshots", [])
            result[phase]["kv_cache"] = {
                "total_calls": kv["total_calls"],
                "total_blocks_allocated": kv["total_blocks_allocated"],
                "avg_blocks_per_call": kv["total_blocks_allocated"] / kv["total_calls"],
                "final_free_blocks": snaps[-1] if snaps else 0,
                "min_free_blocks": min(snaps) if snaps else 0,
                "avg_free_blocks": sum(snaps) / len(snaps) if snaps else 0,
                "total_computed_tokens": kv.get("total_computed_tokens", 0),
                "total_external_computed_tokens": kv.get("total_external_computed_tokens", 0),
            }
        else:
            result[phase]["kv_cache"] = None

    return result