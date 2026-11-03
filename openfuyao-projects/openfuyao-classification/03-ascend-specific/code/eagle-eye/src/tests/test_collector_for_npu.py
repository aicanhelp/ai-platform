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
This module unit-tests NPUCollector's DCMI-backed helpers for device utilization rate,
temperature, health state, and aggregated error codes.
"""

import unittest
import ctypes

from ctypes import c_int, c_uint
from unittest.mock import patch, Mock

from common.npu_metrics.constants import (RET_NO_FUNCTION, RET_INVALID_VALUE, INVALID_TEMPERATURE,
                                          MAX_ERROR_CODE_COUNT, DEVICE_NOT_READY_ERROR_CODE,
                                          CARD_DROP_FAULT_CODE, AICORE_CURRENT_FREQ,
                                          AICORE_RATED_FREQ, NETWORK_SUCCESS)
from hardware_monitor.collector.metrics.npu.collector_for_npu import NPUCollector

class TestGetDeviceAllNpuInfo(unittest.TestCase):
    """
    TestGetDeviceAllNpuInfo tests NPUCollector._get_device_all_npu_info for
    aggregated device info: success path (all sub-calls succeed) and partial
    failure path (some sub-calls return errors).
    """

    def test_success(self):
        """
        Tests _get_device_all_npu_info when all sub-calls succeed; result
        contains base device info and all NPU metric fields.
        """
        base = {
            "node_name": "node0",
            "phy_id": 0,
            "card_id": 0,
            "device_id": 0,
            "namespace": "NA",
            "pod_name": "NA",
            "container_name": "NA",
            "npu_chip_info_aicore_utilization_rate": "NA",
            "npu_chip_info_temperature": "NA",
            "npu_chip_info_aicore_current_freq": "NA",
            "npu_chip_info_aicore_rated_freq": "NA",
            "npu_chip_info_health_status": "NA",
            "npu_chip_info_error_code": "NA",
            "npu_chip_info_network_status": "NA",
            "errors": [],
        }

        with patch("hardware_monitor.collector.metrics.npu.collector_for_npu.init_device_info",
                   return_value=dict(base)):
            with patch.object(NPUCollector, "_get_device_utilization_rate",
                             return_value={"utilization_rate": 50}):
                with patch.object(NPUCollector, "_get_device_temperature",
                                 return_value={"temperature": 60}):
                    with patch.object(NPUCollector, "_get_device_frequency",
                                     side_effect=[{"frequency": 1300}, {"frequency": 1400}]):
                        with patch.object(NPUCollector, "_get_device_health",
                                         return_value={"health": 0}):
                            with patch.object(NPUCollector, "_get_device_all_error_code",
                                             return_value={"error_codes": []}):
                                with patch.object(NPUCollector, "_get_device_network_status",
                                                 return_value={"health": NETWORK_SUCCESS}):
                                    # pylint: disable=protected-access
                                    res = NPUCollector._get_device_all_npu_info(
                                        node_name="node0", logic_id=0, phy_id=0,
                                        card_id=0, device_id=0
                                    )

        self.assertEqual(res["node_name"], "node0")
        self.assertEqual(res["phy_id"], 0)
        self.assertEqual(res["card_id"], 0)
        self.assertEqual(res["device_id"], 0)
        self.assertEqual(res["npu_chip_info_aicore_utilization_rate"], 50)
        self.assertEqual(res["npu_chip_info_temperature"], 60)
        self.assertEqual(res["npu_chip_info_aicore_current_freq"], 1300)
        self.assertEqual(res["npu_chip_info_aicore_rated_freq"], 1400)
        self.assertEqual(res["npu_chip_info_health_status"], 0)
        self.assertEqual(res["npu_chip_info_error_code"], [])
        self.assertEqual(res["npu_chip_info_network_status"], 0)
        self.assertEqual(res["errors"], [])

    def test_partial_errors(self):
        """
        Tests _get_device_all_npu_info when some sub-calls fail; result
        has base info, success values where mocked, and errors list populated.
        """
        base = {
            "node_name": "master",
            "phy_id": 1,
            "card_id": 0,
            "device_id": 0,
            "namespace": "NA",
            "pod_name": "NA",
            "container_name": "NA",
            "npu_chip_info_aicore_utilization_rate": "NA",
            "npu_chip_info_temperature": "NA",
            "npu_chip_info_aicore_current_freq": "NA",
            "npu_chip_info_aicore_rated_freq": "NA",
            "npu_chip_info_health_status": "NA",
            "npu_chip_info_error_code": "NA",
            "npu_chip_info_network_status": "NA",
            "errors": [],
        }

        with patch("hardware_monitor.collector.metrics.npu.collector_for_npu.init_device_info",
                   return_value=dict(base)):
            with patch.object(NPUCollector, "_get_device_utilization_rate",
                             return_value={"error_code": RET_NO_FUNCTION,
                                          "error_message": "util rate failed"}):
                with patch.object(NPUCollector, "_get_device_temperature",
                                 return_value={"temperature": 55}):
                    with patch.object(NPUCollector, "_get_device_frequency",
                                     side_effect=[{"frequency": 1200}, {"frequency": 1400}]):
                        with patch.object(NPUCollector, "_get_device_health",
                                         return_value={"error_code": -8001,
                                                       "error_message": "health failed"}):
                            with patch.object(NPUCollector, "_get_device_all_error_code",
                                             return_value={"error_codes": [100]}):
                                with patch.object(NPUCollector, "_get_device_network_status",
                                                 return_value={"error_code": RET_NO_FUNCTION,
                                                               "error_message": "network not available"}):
                                    # pylint: disable=protected-access
                                    res = NPUCollector._get_device_all_npu_info(
                                        node_name="master", logic_id=1, phy_id=1,
                                        card_id=0, device_id=0
                                    )

        self.assertEqual(res["node_name"], "master")
        self.assertEqual(res["phy_id"], 1)
        self.assertEqual(res["npu_chip_info_temperature"], 55)
        self.assertEqual(res["npu_chip_info_aicore_current_freq"], 1200)
        self.assertEqual(res["npu_chip_info_aicore_rated_freq"], 1400)
        self.assertEqual(res["npu_chip_info_error_code"], [100])
        self.assertIn("npu_chip_info_aicore_utilization_rate", res)
        self.assertIn("npu_chip_info_health_status", res)
        self.assertIn("npu_chip_info_network_status", res)
        self.assertEqual(len(res["errors"]), 3)
        error_codes = [e["error_code"] for e in res["errors"]]
        self.assertIn(RET_NO_FUNCTION, error_codes)
        self.assertIn(-8001, error_codes)

class TestGetDeviceFrequency(unittest.TestCase):
    """
    TestGetDeviceFrequency tests `_get_device_frequency` for symbol presence,
    error returns, value range checks, and success.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_func_missing(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_frequency behavior
        when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = NPUCollector._get_device_frequency(
            logic_id=0, card_id=0, device_id=0, dev_type=AICORE_CURRENT_FREQ
        )
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])
    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_ret_error(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_frequency behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_frequency.return_value = -8001

        # pylint: disable=protected-access
        res = NPUCollector._get_device_frequency(
            logic_id=1, card_id=0, device_id=0, dev_type=AICORE_CURRENT_FREQ
        )
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("Failed to get frequency for logicID(1)", res["error_message"])
        self.assertIn("AICore Current", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_frequency_invalid(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_frequency behavior
        when the frequency value is >= MAX_INT32.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake(card, dev, freq_type, freq_ref):  # pylint: disable=unused-argument
            ctypes.cast(freq_ref, ctypes.POINTER(c_uint)).contents.value = 2 ** 31  # MAX_INT32
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_frequency.side_effect = fake

        # pylint: disable=protected-access
        res = NPUCollector._get_device_frequency(
            logic_id=0, card_id=0, device_id=0, dev_type=AICORE_CURRENT_FREQ
        )
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("out of range", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_frequency behavior
        when the frequency value is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake(card, dev, freq_type, freq_ref):  # pylint: disable=unused-argument
            ctypes.cast(freq_ref, ctypes.POINTER(c_uint)).contents.value = 1300
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_frequency.side_effect = fake

        # pylint: disable=protected-access
        res = NPUCollector._get_device_frequency(
            logic_id=0, card_id=0, device_id=0, dev_type=AICORE_CURRENT_FREQ
        )
        self.assertEqual(res, {"frequency": 1300})

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_rated_freq(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_frequency behavior
        when the rated frequency is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake(card, dev, freq_type, freq_ref):  # pylint: disable=unused-argument
            ctypes.cast(freq_ref, ctypes.POINTER(c_uint)).contents.value = 1400
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_frequency.side_effect = fake

        # pylint: disable=protected-access
        res = NPUCollector._get_device_frequency(
            logic_id=0, card_id=0, device_id=0, dev_type=AICORE_RATED_FREQ
        )
        self.assertEqual(res, {"frequency": 1400})


class TestGetDeviceUtilizationRate(unittest.TestCase):
    """
    TestGetDeviceUtilizationRate tests NPUCollector._get_device_utilization_rate
    for success and error paths (missing function / error return).
    """

    def test_success(self):
        """
        Tests _get_device_utilization_rate when utilization rate is retrieved successfully.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_npu.get_utilization_rate",
                   return_value=(75, None)):
            # pylint: disable=protected-access
            res = NPUCollector._get_device_utilization_rate(logic_id=0, card_id=0, device_id=0)
            self.assertEqual(res, {"utilization_rate": 75})

    def test_error(self):
        """
        Tests _get_device_utilization_rate when get_utilization_rate returns an error.
        """
        with patch("hardware_monitor.collector.metrics.npu.collector_for_npu.get_utilization_rate",
                   return_value=(RET_NO_FUNCTION,
                                 RuntimeError("dcmi_get_device_utilization_rate function not found "
                                              "in dcmi shared object"))):
            # pylint: disable=protected-access
            res = NPUCollector._get_device_utilization_rate(logic_id=1, card_id=0, device_id=0)
            self.assertEqual(res["error_code"], RET_NO_FUNCTION)
            self.assertIn("Failed to get chip info from logicID(1)", res["error_message"])
            self.assertIn("not found", res["error_message"])

class TestGetDeviceTemperature(unittest.TestCase):
    """
    TestGetDeviceTemperature tests `_get_device_temperature` for symbol presence,
    error returns, value checks, and success.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_func_missing(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_temperature behavior
        when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = NPUCollector._get_device_temperature(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_ret_error(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_temperature behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_temperature.return_value = -8001

        # pylint: disable=protected-access
        res = NPUCollector._get_device_temperature(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("Failed to get temperature for logicID(0)", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_temp_invalid(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_temperature behavior
        when the temperature value is invalid.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake(card, dev, temp_ref):  # pylint: disable=unused-argument
            ctypes.cast(temp_ref, ctypes.POINTER(c_int)).contents.value = INVALID_TEMPERATURE - 1
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_temperature.side_effect = fake

        # pylint: disable=protected-access
        res = NPUCollector._get_device_temperature(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("Invalid temperature value for logicID(0)", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_temperature behavior
        when the temperature value is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake(card, dev, temp_ref):  # pylint: disable=unused-argument
            ctypes.cast(temp_ref, ctypes.POINTER(c_int)).contents.value = 67
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_temperature.side_effect = fake

        # pylint: disable=protected-access
        res = NPUCollector._get_device_temperature(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res, {"temperature": 67})


class TestGetDeviceHealth(unittest.TestCase):
    """
    TestGetDeviceHealth tests `_get_device_health` for symbol presence, error handling,
    flag validation, and success mapping
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_func_missing(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_health behavior
        when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = NPUCollector._get_device_health(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_ret_error(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_health behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_health.return_value = -8001

        # pylint: disable=protected-access
        res = NPUCollector._get_device_health(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("ret: -8001", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_health_invalid_flag(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_health behavior
        when an invalid health flag is returned.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_ok(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 2 ** 31 - 1
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_ok

        # pylint: disable=protected-access
        res = NPUCollector._get_device_health(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("get wrong health state", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_health_success_zero(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_health behavior
        when the health flag is 0 (indicating healthy device).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_ok(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_ok

        # pylint: disable=protected-access
        res = NPUCollector._get_device_health(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res, {"health": 0})

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_health_success_nonzero(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_health behavior
        when the health flag is non-zero (indicating faulty device).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_ok(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 3
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_ok

        # pylint: disable=protected-access
        res = NPUCollector._get_device_health(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res, {"health": 1})


class TestGetDeviceAllErrorCode(unittest.TestCase):
    """
    TestGetDeviceAllErrorCode tests `_get_device_all_error_code` for error-paths, health interplay,
    count validation, and success.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_func_missing(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_all_error_code behavior
        when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None

        # pylint: disable=protected-access
        res = NPUCollector._get_device_all_error_code(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_NO_FUNCTION)
        self.assertIn("not available", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_ret_error(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_all_error_code behavior
        when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_errcodes(card, dev, err_cnt_ref, code_arr, max_cnt):  # pylint: disable=unused-argument
            return -8001

        def fake_health(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_errorcode_v2.side_effect = fake_errcodes
        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_health

        # pylint: disable=protected-access
        res = NPUCollector._get_device_all_error_code(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], -8001)
        self.assertIn("Failed to obtain the device error code", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_device_not_ready_card_drop(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_all_error_code behavior
        when the device health indicates a card drop fault.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_errcodes(card, dev, err_cnt_ref, code_arr, max_cnt):  # pylint: disable=unused-argument
            ctypes.cast(err_cnt_ref, ctypes.POINTER(c_int)).contents.value = 0
            return 0

        def fake_health(card, dev, health_ref):  # pylint: disable=unused-argument
            return DEVICE_NOT_READY_ERROR_CODE

        mock_dcmi_instance.dcmi.dcmi_get_device_errorcode_v2.side_effect = fake_errcodes
        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_health

        # pylint: disable=protected-access
        res = NPUCollector._get_device_all_error_code(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_count"], 1)
        self.assertEqual(res["error_codes"], [int(CARD_DROP_FAULT_CODE)])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_error_count_invalid(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_all_error_code behavior
        when the error count exceeds the maximum allowed.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_errcodes(card, dev, err_cnt_ref, code_arr, max_cnt):  # pylint: disable=unused-argument
            ctypes.cast(err_cnt_ref, ctypes.POINTER(c_int)).contents.value = MAX_ERROR_CODE_COUNT + 1
            return 0

        def fake_health(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_errorcode_v2.side_effect = fake_errcodes
        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_health

        # pylint: disable=protected-access
        res = NPUCollector._get_device_all_error_code(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_code"], RET_INVALID_VALUE)
        self.assertIn("invalid error code count", res["error_message"])

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests NPUCollector._get_device_all_error_code behavior
        when device error codes are successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_errcodes(card, dev, err_cnt_ref, code_arr, max_cnt):  # pylint: disable=unused-argument
            ctypes.cast(err_cnt_ref, ctypes.POINTER(c_int)).contents.value = 3
            code_arr[0] = 100
            code_arr[1] = 200
            code_arr[2] = 0
            return 0

        def fake_health(card, dev, health_ref):  # pylint: disable=unused-argument
            ctypes.cast(health_ref, ctypes.POINTER(c_uint)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_errorcode_v2.side_effect = fake_errcodes
        mock_dcmi_instance.dcmi.dcmi_get_device_health.side_effect = fake_health

        # pylint: disable=protected-access
        res = NPUCollector._get_device_all_error_code(logic_id=0, card_id=0, device_id=0)
        self.assertEqual(res["error_count"], 3)
        self.assertEqual(res["error_codes"], [100, 200])


if __name__ == "__main__":
    unittest.main()
