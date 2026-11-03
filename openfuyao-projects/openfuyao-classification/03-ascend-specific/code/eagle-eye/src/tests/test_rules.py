# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module contains unit tests for NPU detector rules.
"""

from unittest import TestCase
from unittest.mock import patch

from hardware_diagnosis.event.base import FaultType
from hardware_diagnosis.detector.npu.rules import (
    HBMErrorRule, DDRErrorRule, NPUErrorRules, NetworkErrorRule
)
from hardware_diagnosis.event.hardware_event import NPUHardwareEvent


class TestErrorRule(TestCase):
    """
    TestErrorRule verifies the detection of HBM-related errors, such as over-temperature
    and errors from event logs. It checks if the rule correctly creates NPUHardwareEvent
    instances with the appropriate details.
    """
    def test_hbmerror_rule(self):
        """
        Tests the HBMErrorRule's detection of HBM errors, including over-temperature and
        error message events. The test ensures that the error code, fault type, and details
        are correctly set in the resulting NPUHardwareEvent.
        """
        data = {"collector.npu.hbm": {"0": {"node_name": "node",
                                            "phy_id": 0, "card_id": 0, "device_id": 0,
                                            "namespace": "", "pod_name": "", "container_name": "",
                                            "npu_chip_info_hbm_total_memory": 100,
                                            "npu_chip_info_hbm_frequency": 200,
                                            "npu_chip_info_hbm_used_memory": 10,
                                            "npu_chip_info_hbm_temperature": 100,
                                            "npu_chip_info_hbm_bandwidth_utilization": 2,
                                            "npu_chip_info_hbm_utilization": 2,
                                            "errors": [{"error_code": 8008, "error_message": "mock error"}]}}}
        out = HBMErrorRule().detect(data)
        temp_want = NPUHardwareEvent(error_code=hex(100), logic_id=0, card_id=0, device_id=0, phy_id=0,
                                     node_name="node", fault_type=FaultType.HARDWARE, hardware_type="NPU_HBM",
                                     severity=1, event_id="1a4d4730-ebdb-4d29-b097-05430a08f82a",
                                     details="HBM temperature is over threshold",
                                     namespace="", pod_name="", container_name="")
        func_want = NPUHardwareEvent(error_code="8008", logic_id=0, card_id=0, device_id=0, phy_id=0,
                                     node_name="node", fault_type=FaultType.HARDWARE, hardware_type="NPU_HBM",
                                     severity=1, event_id="9fb8f9e3-7167-42fe-a10e-0ccbde86c14a", details="mock error",
                                     namespace="", pod_name="", container_name="")
        self.assertEqual(out[0].error_code, temp_want.error_code)
        self.assertEqual(out[0].details, temp_want.details)
        self.assertEqual(out[0].namespace, "")
        self.assertEqual(out[0].pod_name, "")
        self.assertEqual(out[0].container_name, "")
        self.assertEqual(out[1].error_code, func_want.error_code)
        self.assertEqual(out[1].details, func_want.details)

    def test_ddrerror_rule(self):
        """
        Tests the DDRErrorRule's detection of DDR errors, specifically checking if
        error codes and messages from DDR errors are correctly processed and reported.
        """
        data = {"collector.npu.ddr": {"0": {"node_name": "master",
                                            "phy_id": 1, "card_id": 0, "device_id": 0,
                                            "namespace": "", "pod_name": "", "container_name": "",
                                            "npu_chip_info_total_memory": 300,
                                            "npu_chip_info_used_memory": 30,
                                            "npu_chip_info_memory_frequency": 39,
                                            "npu_chip_info_memory_utilization": 40,
                                            "errors": [{"error_code": 1111, "error_message": "mock DDR error"}]}}}
        out = DDRErrorRule().detect(data)
        want = NPUHardwareEvent(error_code="1111", details="mock DDR error", card_id=0, device_id=0, logic_id=0,
                                phy_id=1, node_name="master", fault_type=FaultType.HARDWARE, event_id="",
                                hardware_type="NPU_DDR", namespace="", pod_name="", container_name="")
        self.assertEqual(out[0].error_code, want.error_code)
        self.assertEqual(out[0].details, want.details)
        self.assertEqual(out[0].namespace, "")
        self.assertEqual(out[0].pod_name, "")
        self.assertEqual(out[0].container_name, "")

    @patch("hardware_diagnosis.detector.npu.rules.get_device_error_code_string")
    def test_npuerror_rules(self, mock_info):
        """
        Tests the NPUErrorRules' detection of NPU-related errors. This test ensures that errors
        such as system memory issues are detected and reported with the appropriate error codes
        and details.
        """
        data = {"collector.npu.npu": {"0": {"node_name": "master",
                                            "phy_id": 0, "card_id": 0, "device_id": 0,
                                            "namespace": "", "pod_name": "", "container_name": "",
                                            "npu_chip_info_utilization": 30,
                                            "npu_chip_info_temperature": 50,
                                            "npu_chip_info_health_status": 0,
                                            "npu_chip_info_error_code": [int(0x8C2FA001)],
                                            "npu_chip_info_network_status": 0,
                                            "errors": [{"error_code": 8001, "error_message": "mock NPU error"}]}}}
        mock_info.return_value = ("system out of memory", None, None)
        out = NPUErrorRules().detect(data)
        chip_want = NPUHardwareEvent(hardware_type="NPU", error_code=hex(0x8C2FA001), device_id=0,
                                     card_id=0, logic_id=0, phy_id=0, node_name="master",
                                     fault_type=FaultType.HARDWARE,
                                     details="system out of memory",
                                     namespace="", pod_name="", container_name="")
        func_want = NPUHardwareEvent(hardware_type="NPU", error_code="8001", device_id=0,
                                     card_id=0, logic_id=0, phy_id=0, node_name="master",
                                     fault_type=FaultType.HARDWARE,
                                     details="mock NPU error",
                                     namespace="", pod_name="", container_name="")
        self.assertEqual(out[0].error_code, chip_want.error_code)
        self.assertEqual(out[0].details, chip_want.details)
        self.assertEqual(out[0].namespace, "")
        self.assertEqual(out[0].pod_name, "")
        self.assertEqual(out[0].container_name, "")
        mock_info.assert_called_once()
        self.assertEqual(out[1].error_code, func_want.error_code)
        self.assertEqual(out[1].details, func_want.details)
        self.assertEqual(out[1].namespace, "")
        self.assertEqual(out[1].pod_name, "")
        self.assertEqual(out[1].container_name, "")

    def test_network_error_rule(self):
        """
        Tests the NetworkErrorRule's detection of network-related errors, specifically
        verifying that network errors are correctly processed and reported with appropriate
        error codes and details. Also verifies that namespace, pod_name, container_name
        from metrics are passed through to the event.
        """
        data = {"collector.npu.network": {"0": {"node_name": "master",
                                                "phy_id": 3, "card_id": 0, "device_id": 0,
                                                "namespace": "kube-system",
                                                "pod_name": "npu-pod-0",
                                                "container_name": "npu-container",
                                                "npu_chip_info_bandwidth_tx": 300,
                                                "npu_chip_info_bandwidth_rx": 300,
                                                "npu_chip_info_link_status": 0,
                                                "errors": [{"error_code": 8001,
                                                            "error_message": "mock network error"}]}}}

        out = NetworkErrorRule().detect(data)
        want = NPUHardwareEvent(hardware_type="NPU_Network", error_code="8001",
                                device_id=0, logic_id=0, card_id=0, phy_id=3,
                                node_name="master",
                                fault_type=FaultType.HARDWARE, details="mock network error",
                                namespace="kube-system", pod_name="npu-pod-0", container_name="npu-container")
        self.assertEqual(out[0].error_code, want.error_code)
        self.assertEqual(out[0].details, want.details)
        self.assertEqual(out[0].namespace, "kube-system")
        self.assertEqual(out[0].pod_name, "npu-pod-0")
        self.assertEqual(out[0].container_name, "npu-container")
