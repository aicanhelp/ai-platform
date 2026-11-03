# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Import Hook mechanism for non-invasive vLLM-Ascend tracing.

Uses sys.meta_path to intercept module loading and apply monkey patches.
VLLMAscendTracerFinder is registered directly at Python startup (via
sitecustomize.py) and patches modules as they are imported.
"""

import sys
import importlib
import importlib.abc
import importlib.util
import logging

logger = logging.getLogger("vllm_ascend.tracing")

# Modules that need patching and their corresponding patch functions
MODULES_TO_PATCH = {
    "vllm.entrypoints.openai.engine.serving": "vllm_ascend_tracing.patches.serving:patch_openai_serving",
    "vllm.entrypoints.openai.chat_completion.serving": "vllm_ascend_tracing.patches.serving:patch_chat_serving",
    "vllm.entrypoints.openai.completion.serving": "vllm_ascend_tracing.patches.serving:patch_completion_serving",
    "vllm.entrypoints.openai.responses.serving": "vllm_ascend_tracing.patches.serving:patch_responses_serving",
    "vllm.v1.core.sched.scheduler": "vllm_ascend_tracing.patches.scheduler:patch_scheduler",
    "vllm.v1.core.kv_cache_manager": "vllm_ascend_tracing.patches.kv_cache_manager:patch_kv_cache_manager",
    "vllm.v1.engine.output_processor": "vllm_ascend_tracing.patches.output_processor:patch_output_processor",
    # Core: Add per_request_store_stats field to KVConnectorOutput for serialization
    "vllm.v1.outputs": "vllm_ascend_tracing.patches.outputs:patch_kv_connector_output",
    # Worker-level: AscendStore stats collection
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer": "vllm_ascend_tracing.patches.ascend_store:patch_kv_transfer_threads",
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker": "vllm_ascend_tracing.patches.ascend_store:patch_pool_worker",
    # Worker-level: stats propagation back to scheduler
    "vllm.v1.worker.kv_connector_model_runner_mixin": "vllm_ascend_tracing.patches.kv_connector:patch_kv_connector_mixin",
    "vllm.distributed.kv_transfer.kv_connector.utils": "vllm_ascend_tracing.patches.kv_connector:patch_kv_output_aggregator",
    # vLLM-Ascend: Patch BalanceScheduler for 0.23.0 compatibility
    "vllm_ascend.patch.platform.patch_balance_schedule": "vllm_ascend_tracing.patches.scheduler:patch_balance_scheduler",
}

# Track which modules have been patched to avoid double-patching
_patched_modules: set = set()


def _get_patch_func(patch_path: str):
    """Import and return a patch function from a dotted path like 'module:func'."""
    module_path, func_name = patch_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _apply_patch(module_name: str, module):
    """Apply the corresponding patch to a loaded module."""
    if module_name in _patched_modules:
        logger.info(f"[tracing] _apply_patch: {module_name} already patched, skipping")
        return
    patch_path = MODULES_TO_PATCH.get(module_name)
    if patch_path is None:
        return
    try:
        logger.info(f"[tracing] _apply_patch: patching {module_name} with {patch_path}")
        patch_func = _get_patch_func(patch_path)
        patch_func(module)
        _patched_modules.add(module_name)
        logger.info(f"[tracing] Patched {module_name}")
    except Exception as e:
        logger.error(f"[tracing] Failed to patch {module_name}: {e}")


class VLLMAscendTracerFinder(importlib.abc.MetaPathFinder):
    """Intercepts vLLM module imports and applies tracing patches."""

    def __init__(self):
        """Patch modules that were already loaded before this finder was registered."""
        for module_name in MODULES_TO_PATCH:
            if module_name in sys.modules:
                _apply_patch(module_name, sys.modules[module_name])

    def find_spec(self, fullname, path, target=None):
        """Intercept module imports for modules we need to patch."""
        if fullname not in MODULES_TO_PATCH:
            return None

        # Skip if already patched
        if fullname in _patched_modules:
            return None

        # Temporarily remove ourselves to avoid recursion
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)

        if spec is None:
            return None

        spec.loader = _VLLMAscendLoader(spec.loader, fullname)
        return spec


class _VLLMAscendLoader:
    """Wraps the original loader to apply patches after module execution."""

    def __init__(self, original_loader, module_name):
        self.original_loader = original_loader
        self.module_name = module_name

    def create_module(self, spec):
        if hasattr(self.original_loader, 'create_module'):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        _apply_patch(self.module_name, module)


def register_hooks():
    """Register the import hooks. Called from sitecustomize.py."""
    sys.meta_path.insert(0, VLLMAscendTracerFinder())
    logger.info("[tracing] VLLMAscendTracerFinder registered")
