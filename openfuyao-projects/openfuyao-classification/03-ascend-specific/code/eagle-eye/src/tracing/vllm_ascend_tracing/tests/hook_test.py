# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for vllm_ascend_tracing.hook."""

import sys
import unittest
from unittest.mock import MagicMock, patch as mock_patch

from vllm_ascend_tracing.hook import (
    VLLMAscendTracerFinder, _VLLMAscendLoader, _apply_patch, _get_patch_func,
    register_hooks, _patched_modules,
)


class TestGetPatchFunc(unittest.TestCase):
    def test_valid_path(self):
        func = _get_patch_func("vllm_ascend_tracing.patches.scheduler:patch_scheduler")
        from vllm_ascend_tracing.patches.scheduler import patch_scheduler
        self.assertIs(func, patch_scheduler)

    def test_invalid_module_raises(self):
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            _get_patch_func("nonexistent.module:func")


class TestApplyPatch(unittest.TestCase):
    def setUp(self):
        _patched_modules.clear()

    def test_applies_patch_successfully(self):
        module = MagicMock()
        module.KVCacheManager = MagicMock()
        module.KVCacheManager.allocate_slots = MagicMock()

        _apply_patch("vllm.v1.core.kv_cache_manager", module)
        self.assertIn("vllm.v1.core.kv_cache_manager", _patched_modules)

    def test_skips_already_patched(self):
        _patched_modules.add("some.module")
        module = MagicMock()
        # Should not raise even though module doesn't have the right structure
        _apply_patch("some.module", module)

    def test_skips_unknown_module(self):
        _apply_patch("unknown.module.not.in.list", MagicMock())
        self.assertNotIn("unknown.module.not.in.list", _patched_modules)


class TestVLLMAscendTracerFinder(unittest.TestCase):
    def setUp(self):
        _patched_modules.clear()

    def test_find_spec_returns_none_for_unknown(self):
        finder = VLLMAscendTracerFinder()
        result = finder.find_spec("some.random.module", None)
        self.assertIsNone(result)

    def test_find_spec_returns_none_if_already_patched(self):
        _patched_modules.add("vllm.v1.core.kv_cache_manager")
        finder = VLLMAscendTracerFinder()
        result = finder.find_spec("vllm.v1.core.kv_cache_manager", None)
        self.assertIsNone(result)


class TestVLLMAscendLoader(unittest.TestCase):
    def test_create_module_delegates(self):
        original_loader = MagicMock()
        original_loader.create_module.return_value = "mock_module"
        loader = _VLLMAscendLoader(original_loader, "test.module")
        result = loader.create_module(MagicMock())
        self.assertEqual(result, "mock_module")

    def test_create_module_without_method(self):
        original_loader = MagicMock(spec=[])  # no create_module attr
        loader = _VLLMAscendLoader(original_loader, "test.module")
        result = loader.create_module(MagicMock())
        self.assertIsNone(result)

    def test_exec_module_applies_patch(self):
        original_loader = MagicMock()
        module = MagicMock()

        with mock_patch("vllm_ascend_tracing.hook._apply_patch") as mock_apply:
            loader = _VLLMAscendLoader(original_loader, "test.module")
            loader.exec_module(module)
            original_loader.exec_module.assert_called_once_with(module)
            mock_apply.assert_called_once_with("test.module", module)


class TestRegisterHooks(unittest.TestCase):
    def test_registers_finder(self):
        # Remove any existing finders first
        original_meta_path = list(sys.meta_path)
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, VLLMAscendTracerFinder)]

        register_hooks()

        finders = [f for f in sys.meta_path if isinstance(f, VLLMAscendTracerFinder)]
        self.assertEqual(len(finders), 1)

        # Cleanup
        sys.meta_path = original_meta_path


if __name__ == "__main__":
    unittest.main()
