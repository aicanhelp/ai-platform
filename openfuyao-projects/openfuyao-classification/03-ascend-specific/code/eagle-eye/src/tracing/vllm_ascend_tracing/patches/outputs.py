# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Patch KVConnectorOutput.is_empty() to prevent vllm from discarding outputs
that carry per_request_store_stats (dynamically attached by kv_connector.py)."""

import logging

logger = logging.getLogger("vllm_ascend.tracing")


def patch_kv_connector_output(module):
    """Patch KVConnectorOutput.is_empty() to account for per_request_store_stats."""
    kv_output_cls = module.KVConnectorOutput

    # Check if already patched
    if hasattr(kv_output_cls, '_tracing_patched'):
        return

    # Only patch is_empty() to check for the new field
    original_is_empty = kv_output_cls.is_empty

    def patched_is_empty(self):
        # Check if per_request_store_stats exists and is non-empty
        has_stats = getattr(self, 'per_request_store_stats', None)
        return original_is_empty(self) and not has_stats

    kv_output_cls.is_empty = patched_is_empty
    kv_output_cls._tracing_patched = True

    logger.info("[tracing] Patched KVConnectorOutput.is_empty() to handle per_request_store_stats")
