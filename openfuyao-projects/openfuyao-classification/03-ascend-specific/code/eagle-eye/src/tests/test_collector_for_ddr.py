# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module provides unit tests for DDRCollector, including device basic
information resolution and DDR memory metrics retrieval.
"""

import unittest
import ctypes

from unittest.mock import patch, Mock

from common.npu_metrics.dcmi.dcmi_wrapper import DcmiMemoryInfoStruct
from common.npu_metrics.constants import RET_NO_FUNCTION, RET_INVALID_VALUE, ASCEND_910B
from hardware_monitor.collector.metrics.npu.collector_for_ddr import DDRCollector


class TestGetDeviceBasicInfo(unittest.TestCase):
    """
    TestGetDeviceBasicInfo tests DDRCollector._get_device_basic_info behavior
    """

    def test_resolve_device_type_error(self):
        """
        Tests DDRCollector._get_device_basic_info behavior
        when the device type resolution fails.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_ddr.fetch_device_info_and_type",
                   return_value=({}, {}, None, RET_NO_FUNCTION, RuntimeError("not found"))):
            # pylint: disable=protected-access
            out = DDRCollector._get_device_basic_info(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(out["error_code"], RET_NO_FUNCTION)
            self.assertIn("Failed to resolve device type from logicID(0): not found",
                          out["error_message"])

    def test_device_type_block_910b(self):
        """
        Tests DDRCollector._get_device_basic_info behavior
        when the device type is blocked for ASCEND_910B.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_ddr.fetch_device_info_and_type",
                   return_value=({}, {}, ASCEND_910B, None, None)):
            # pylint: disable=protected-access
            out = DDRCollector._get_device_basic_info(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(out["error_code"], RET_INVALID_VALUE)
            self.assertIn(f"LogicID(0): {ASCEND_910B} does not have ddr module",
                          out["error_message"])

    def test_device_type_ok(self):
        """
        Tests DDRCollector._get_device_basic_info behavior
        when the device type is valid (ASCEND_310B).
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_ddr.fetch_device_info_and_type",
                   return_value=({}, {}, "ASCEND_310B", None, None)):
            # pylint: disable=protected-access
            out = DDRCollector._get_device_basic_info(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(out, {})


class TestGetDeviceMemoryInfo(unittest.TestCase):
    """
    TestGetDeviceMemoryInfo tests DDRCollector._get_device_memory_info behavior.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_func_missing(self, mock_get_dcmi_instance):
        """
        Tests DDRCollector._get_device_memory_info behavior
        when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = DDRCollector._get_device_memory_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_ret_error(self, mock_get_dcmi_instance):
        """
        Tests DDRCollector._get_device_memory_info behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_memory_info_v3.return_value = -8001
        # pylint: disable=protected-access
        res = DDRCollector._get_device_memory_info(logic_id=-1, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("ret: -8001", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_invalid_total_zero(self, mock_get_dcmi_instance):
        """
        Tests DDRCollector._get_device_memory_info behavior
        when the total memory size is zero.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiMemoryInfoStruct)).contents
            s.memory_size = 0
            s.memory_available = 1
            s.freq = 1
            s.hugepagesize = 1
            s.hugepages_total = 1
            s.hugepages_free = 1
            s.utiliza = 1
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_memory_info_v3.side_effect = fake_func

        # pylint: disable=protected-access
        res = DDRCollector._get_device_memory_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("Invalid memory info", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_invalid_available_gt_total(self, mock_get_dcmi_instance):
        """
        Tests DDRCollector._get_device_memory_info behavior
        when the available memory exceeds the total memory.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiMemoryInfoStruct)).contents
            s.memory_size = 100
            s.memory_available = 200
            s.freq = 1600
            s.hugepagesize = 2048
            s.hugepages_total = 100
            s.hugepages_free = 40
            s.utiliza = 50
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_memory_info_v3.side_effect = fake_func

        # pylint: disable=protected-access
        res = DDRCollector._get_device_memory_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("Invalid memory info", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests DDRCollector._get_device_memory_info behavior
        when memory info is retrieved successfully.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiMemoryInfoStruct)).contents
            s.memory_size = 1024
            s.memory_available = 256
            s.freq = 1600
            s.hugepagesize = 2097152
            s.hugepages_total = 512
            s.hugepages_free = 128
            s.utiliza = 25
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_memory_info_v3.side_effect = fake_func

        # pylint: disable=protected-access
        res = DDRCollector._get_device_memory_info(logic_id=0, card_id=0, device_id=0)
        self.assertNotIn("error_code", res)
        self.assertEqual(res["memory_size"], 1024)
        self.assertEqual(res["memory_available"], 256)
        self.assertEqual(res["freq"], 1600)
        self.assertEqual(res["hugepagesize"], 2097152)
        self.assertEqual(res["hugepages_total"], 512)
        self.assertEqual(res["hugepages_free"], 128)
        self.assertEqual(res["utiliza"], 25)


if __name__ == "__main__":
    unittest.main()
