# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patches for KV connector output pipeline.

patch_kv_connector_mixin: Attaches per-request store stats from Worker
to KVConnectorOutput (piggybacking on ModelRunnerOutput).

patch_kv_output_aggregator: Aggregates store stats across TP workers
(take max duration since TP is parallel, bottleneck = slowest rank).
"""

import functools
import logging
from contextlib import contextmanager

logger = logging.getLogger("vllm_ascend.tracing")


def patch_kv_connector_mixin(module):
    """Patch _get_kv_connector_output to attach store stats to output."""
    mixin_cls = module.KVConnectorModelRunnerMixin
    original_get_output = mixin_cls._get_kv_connector_output

    @contextmanager
    def patched_get_kv_connector_output(scheduler_output, wait_for_save=True,
                                        defer_finalize=False):
        with original_get_output(scheduler_output, wait_for_save,
                                 defer_finalize) as output:
            yield output

        # After context manager exits, try to collect store stats from worker.
        # Stats accumulate in the worker until collected. If save is async and
        # hasn't completed yet, stats will be empty here but available on the
        # next forward step's collection.
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        kv_connector = get_kv_transfer_group()

        worker = _find_worker_with_stats(kv_connector)
        if worker is not None:
            store_stats = worker.get_per_request_store_stats()
            if store_stats:
                output.per_request_store_stats = store_stats
        else:
            logger.warning_once("[tracing] No worker with store stats found in connector %s",
                                type(kv_connector).__name__)

    mixin_cls._get_kv_connector_output = staticmethod(patched_get_kv_connector_output)


def patch_kv_output_aggregator(module):
    """Patch KVOutputAggregator.aggregate to preserve per_request_store_stats."""
    aggregator_cls = module.KVOutputAggregator
    logger.debug("[tracing] Patching %s.aggregate", aggregator_cls.__name__)
    original_aggregate = aggregator_cls.aggregate

    # Patch from_connector to verify aggregator is initialized
    original_from_connector = aggregator_cls.from_connector

    @staticmethod
    def patched_from_connector(connector, world_size):
        logger.debug("[tracing] KVOutputAggregator.from_connector called with world_size=%d", world_size)
        return original_from_connector(connector, world_size)

    aggregator_cls.from_connector = patched_from_connector

    @functools.wraps(original_aggregate)
    def patched_aggregate(self, outputs, output_rank=0):
        # Collect per_request_store_stats from worker outputs before aggregation
        aggregated_store_stats = {}

        for idx, model_runner_output in enumerate(outputs):
            if model_runner_output is None:
                continue
            kv_output = model_runner_output.kv_connector_output
            if kv_output is None:
                continue

            # Read from the official field (now serializable via outputs.py patch)
            worker_stats = getattr(kv_output, 'per_request_store_stats', None)
            if worker_stats:
                for req_id, stats in worker_stats.items():
                    if req_id not in aggregated_store_stats:
                        aggregated_store_stats[req_id] = dict(stats)
                    else:
                        _merge_tp_stats(aggregated_store_stats[req_id], stats)

        # Call original aggregate
        result = original_aggregate(self, outputs, output_rank)

        # Attach aggregated store stats to result
        if result is not None and aggregated_store_stats:
            if hasattr(result, 'kv_connector_output') and result.kv_connector_output is not None:
                result.kv_connector_output.per_request_store_stats = aggregated_store_stats
                logger.debug("[tracing] aggregator: attached store_stats for %d requests", len(aggregated_store_stats))
            else:
                logger.warning_once("[tracing] aggregator: result has no kv_connector_output, cannot attach stats")

        return result

    aggregator_cls.aggregate = patched_aggregate


def _find_worker_with_stats(kv_connector):
    """Find the worker with per_request_store_stats from a connector (or MultiConnector)."""
    worker = getattr(kv_connector, 'connector_worker', None)
    if worker is not None and hasattr(worker, 'get_per_request_store_stats'):
        return worker
    # MultiConnector: search sub-connectors
    if hasattr(kv_connector, '_connectors'):
        for c in kv_connector._connectors:
            w = getattr(c, 'connector_worker', None)
            if w is not None and hasattr(w, 'get_per_request_store_stats'):
                return w
    return None


def _merge_tp_stats(existing, new_stats):
    """Merge stats from another TP rank (max duration, max calls)."""
    for op in ("save", "load"):
        existing[f"{op}_duration_ms"] = max(
            existing.get(f"{op}_duration_ms", 0),
            new_stats.get(f"{op}_duration_ms", 0),
        )
        existing[f"{op}_max_duration_ms"] = max(
            existing.get(f"{op}_max_duration_ms", 0),
            new_stats.get(f"{op}_max_duration_ms", 0),
        )
        existing[f"{op}_calls"] = max(
            existing.get(f"{op}_calls", 0),
            new_stats.get(f"{op}_calls", 0),
        )
