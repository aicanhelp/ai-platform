# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""sitecustomize.py - Auto-executed by Python on startup.

Registers import hooks for vLLM-Ascend tracing instrumentation.
This file should be placed in the Python site-packages directory.
"""

import os
import logging

# Only activate if tracing is explicitly enabled
if os.environ.get("VLLM_ASCEND_TRACING_ENABLED", "").lower() in ("1", "true"):
    log_level = os.environ.get("VLLM_ASCEND_TRACING_LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Only configure the vllm_ascend.tracing logger, not the root logger,
    # to avoid interfering with vLLM / third-party library log behaviour.
    logger = logging.getLogger("vllm_ascend.tracing")
    logger.setLevel(numeric_level)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setLevel(numeric_level)
        _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(_handler)
    logger.propagate = False
    logger.info("[tracing] Initializing vLLM-Ascend tracing hooks (log_level=%s)", log_level)

    try:
        from vllm_ascend_tracing.hook import register_hooks
        register_hooks()
    except Exception as e:
        logger.error(f"[tracing] Failed to register hooks: {e}")
