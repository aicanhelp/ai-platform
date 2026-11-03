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
This module provides unit tests for HBMCollector, including collection of
HBM metrics, device information retrieval, and utilization rate handling.
"""

import unittest
import ctypes

from unittest.mock import patch, Mock
from typing import Dict, Any

from common.npu_metrics.dcmi.dcmi_wrapper import DcmiHBMInfoStruct
from common.npu_metrics.constants import RET_INVALID_VALUE, RET_NO_FUNCTION
from hardware_monitor.collector.metrics.npu.collector_for_hbm import HBMCollector

def _init_result_with_fields() -> Dict[str, Any]:
    return {
        "phy_id": "NA",
        "card_id": "NA",
        "device_id": "NA",
        "namespace": "NA",
        "pod_name": "NA",
        "container_name": "NA",
        "npu_chip_info_hbm_total_memory": "NA",
        "npu_chip_info_hbm_frequency": "NA",
        "npu_chip_info_hbm_used_memory": "NA",
        "npu_chip_info_hbm_temperature": "NA",
        "npu_chip_info_hbm_bandwidth_utilization": "NA",
        "npu_chip_info_hbm_utilization": "NA",
        "errors": [],
    }


class TestGetDeviceAllHBMInfo(unittest.TestCase):
    """
    TestGetDeviceAllHBMInfo tests HBMCollector._get_device_all_hbm_info behavior.
    """

    def test_hbm_ok_util_ok(self):
        """
        Tests HBMCollector._get_device_all_hbm_info behavior
        when both HBM utilization rate and HBM info retrieval are successful.
        """
        with patch(
                "hardware_monitor.collector.metrics.npu.collector_for_hbm.HBMCollector._get_device_hbm_utilization_rate",
                return_value={"utilization_rate": 42}), \
                patch("hardware_monitor.collector.metrics.npu.collector_for_hbm.HBMCollector._get_device_hbm_info",
                      return_value={
                          "memory_size": 123456,
                          "frequency": 1600,
                          "memory_usage": 7890,
                          "temperature": 65,
                          "bandwidth_util_rate": 88,
                      }), \
                patch("hardware_monitor.collector.metrics.npu.collector_for_hbm.HBMCollector._get_device_basic_info",
                      return_value={}), \
                patch("hardware_monitor.collector.metrics.npu.collector_for_hbm.init_device_info",
                      return_value=(_init_result_with_fields() | {"node_name": "master",
                                                                  "phy_id": 0,
                                                                  "card_id": 0, "device_id": 0})):
            # pylint: disable=protected-access
            out = HBMCollector._get_device_all_hbm_info(node_name="master", logic_id=0, phy_id=0, card_id=0,
                                                        device_id=0)

            expected = _init_result_with_fields()

            expected.update({
                "node_name": "master",
                "phy_id": 0,
                "card_id": 0,
                "device_id": 0,
                "namespace": "NA",
                "pod_name": "NA",
                "container_name": "NA",
                "npu_chip_info_hbm_total_memory": 123456,
                "npu_chip_info_hbm_frequency": 1600,
                "npu_chip_info_hbm_used_memory": 7890,
                "npu_chip_info_hbm_temperature": 65,
                "npu_chip_info_hbm_bandwidth_utilization": 88,
                "npu_chip_info_hbm_utilization": 42,
            })
            self.assertEqual(out, expected)


class TestGetDeviceHBMInfo(unittest.TestCase):
    """
    TestGetDeviceHBMInfo tests HBMCollector._get_device_hbm_info behavior.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests HBMCollector._get_device_hbm_info behavior when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = HBMCollector._get_device_hbm_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests HBMCollector._get_device_hbm_info behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_hbm_info.return_value = -8001
        # pylint: disable=protected-access
        res = HBMCollector._get_device_hbm_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("failed", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_temp(self, mock_get_dcmi_instance):
        """
        Tests HBMCollector._get_device_hbm_info behavior
        when the temperature value is invalid.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiHBMInfoStruct)).contents
            s.memory_size = 123
            s.freq = 1600
            s.memory_usage = 45
            s.temp = -10
            s.bandwith_util_rate = 88
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_hbm_info.side_effect = fake_func

        # pylint: disable=protected-access
        res = HBMCollector._get_device_hbm_info(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("Invalid HBM temperature", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests HBMCollector._get_device_hbm_info behavior
        when memory info is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiHBMInfoStruct)).contents
            s.memory_size = 987654321
            s.freq = 1333
            s.memory_usage = 5555
            s.temp = 67
            s.bandwith_util_rate = 92
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_hbm_info.side_effect = fake_func

        # pylint: disable=protected-access
        res = HBMCollector._get_device_hbm_info(logic_id=0, card_id=0, device_id=0)
        self.assertNotIn("error_code", res)
        self.assertEqual(res["memory_size"], 987654321)
        self.assertEqual(res["frequency"], 1333)
        self.assertEqual(res["memory_usage"], 5555)
        self.assertEqual(res["temperature"], 67)
        self.assertEqual(res["bandwidth_util_rate"], 92)


class TestGetDeviceHBMUtilizationRate(unittest.TestCase):
    """
    TestGetDeviceHBMUtilizationRate tests HBMCollector._get_device_hbm_utilization_rate behavior.
    """

    def test_success(self):
        """
        Tests HBMCollector._get_device_hbm_utilization_rate behavior
        when utilization rate retrieval is successful.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_hbm.get_utilization_rate",
                   return_value=(99, None)):
            # pylint: disable=protected-access
            res = HBMCollector._get_device_hbm_utilization_rate(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(res, {"utilization_rate": 99})

    def test_error(self):
        """
        Tests HBMCollector._get_device_hbm_utilization_rate behavior
        when there is an error in retrieving the utilization rate.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_hbm.get_utilization_rate",
                   return_value=(RET_NO_FUNCTION, RuntimeError("dcmi_get_device_utilization_rate function not found "
                                                               "in dcmi shared object"))):
            # pylint: disable=protected-access
            res = HBMCollector._get_device_hbm_utilization_rate(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(res["error_code"], RET_NO_FUNCTION)
            self.assertIn("Failed to get hbm utilization rate from logicID(0)", res["error_message"])
            self.assertIn("not found", res["error_message"])


if __name__ == "__main__":
    unittest.main()
