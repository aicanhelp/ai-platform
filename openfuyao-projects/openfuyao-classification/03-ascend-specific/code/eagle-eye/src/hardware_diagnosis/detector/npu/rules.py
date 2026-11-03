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
This module defines NPU collector error detection rules, including HBM, DDR,
NPU core, and network error rules. Each rule collects collector metrics via
MainCollector, evaluates them against thresholds or error codes, and generates
HardwareEvent instances for further fault handling and recovery.
"""

import logging
import os

from collections import deque
from typing import Any, Dict, List, Optional

from hardware_diagnosis.detector.base import ErrorRule
from hardware_diagnosis.event.hardware_event import NPUHardwareEvent
from hardware_diagnosis.event.base import FaultType
from hardware_diagnosis.recovery.hardware_strategies.npu.utils import get_device_error_code_string
from hardware_diagnosis.constant.constant import HBM_TEMPERATURE_THRESHOLD, NPU_CHIP_TEMPERATURE_THRESHOLD

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)

# Number of samples to collect before checking frequency/temperature rule
FREQ_TEMP_HISTORY_SIZE = 4


def _container_kwargs(metrics: dict) -> dict:
    """namespace, pod_name, container_name from metrics for event creation."""
    return {
        "namespace": metrics.get("namespace"),
        "pod_name": metrics.get("pod_name"),
        "container_name": metrics.get("container_name"),
    }


class HBMErrorRule(ErrorRule):
    """
    HBMErrorRule detects HBM-related errors such as over-temperature and reported error events.
    """

    def __init__(self):
        super().__init__("NPU_HBM")
        self.hbm_resource = []

    def error_code(self, event: dict) -> str:
        return str(event["error_code"])

    def detect(self, data: dict) -> Optional[List[NPUHardwareEvent]]:
        events = []
        self.hbm_resource = []
        hbm_data = data.get("collector.npu.hbm", {})
        for logic_id, metrics in hbm_data.items():
            node_name = metrics.get("node_name")
            card_id = metrics.get("card_id")
            device_id = metrics.get("device_id")
            phy_id = metrics.get("phy_id")
            self.hbm_resource.append(metrics)
            temperature = metrics.get("npu_chip_info_hbm_temperature")
            extra = _container_kwargs(metrics)
            if (temperature != "NA" and int(temperature)) > HBM_TEMPERATURE_THRESHOLD:
                event = NPUHardwareEvent(hardware_type=self.name,
                                         error_code=hex(100),
                                         node_name=node_name,
                                         card_id=card_id,
                                         device_id=device_id,
                                         logic_id=logic_id,
                                         phy_id=phy_id,
                                         fault_type=FaultType.HARDWARE,
                                         details="HBM temperature is over threshold",
                                         **extra)
                events.append(event)

            if metrics.get("errors"):
                for error_event in metrics["errors"]:
                    event = NPUHardwareEvent(hardware_type=self.name,
                                             error_code=self.error_code(error_event),
                                             node_name=node_name,
                                             card_id=card_id,
                                             device_id=device_id,
                                             logic_id=logic_id,
                                             phy_id=phy_id,
                                             fault_type=FaultType.HARDWARE,
                                             details=error_event["error_message"],
                                             **extra)
                    events.append(event)
        return events


class DDRErrorRule(ErrorRule):
    """
    DDRErrorRule detects DDR-related error events from collected metrics.
    """

    def __init__(self):
        super().__init__("NPU_DDR")
        self.ddr_resource = []

    def error_code(self, event) -> str:
        return str(event["error_code"])

    def detect(self, data: dict) -> Optional[List[NPUHardwareEvent]]:
        events = []
        self.ddr_resource = []
        ddr_data = data.get("collector.npu.ddr", {})
        for logic_id, metrics in ddr_data.items():
            node_name = metrics.get("node_name")
            device_id = metrics.get("device_id")
            card_id = metrics.get("card_id")
            phy_id = metrics.get("phy_id")
            self.ddr_resource.append(metrics)
            extra = _container_kwargs(metrics)
            if metrics.get("errors"):
                for error_event in metrics["errors"]:
                    event = NPUHardwareEvent(hardware_type=self.name,
                                             error_code=self.error_code(error_event),
                                             node_name=node_name,
                                             card_id=card_id,
                                             device_id=device_id,
                                             logic_id=logic_id,
                                             phy_id=phy_id,
                                             fault_type=FaultType.HARDWARE,
                                             details=error_event["error_message"],
                                             **extra)
                    events.append(event)
        return events


class NetworkErrorRule(ErrorRule):
    """
    NetworkErrorRule detects network-related errors reported by NPU metrics.
    """

    def __init__(self):
        super().__init__("NPU_Network")
        self.network_resource = []

    def error_code(self, event) -> str:
        return str(event["error_code"])

    def detect(self, data: dict) -> Optional[List[NPUHardwareEvent]]:
        events = []
        self.network_resource = []
        network_data = data.get("collector.npu.network", {})
        for logic_id, metrics in network_data.items():
            node_name = metrics.get("node_name")
            device_id = metrics.get("device_id")
            card_id = metrics.get("card_id")
            phy_id = metrics.get("phy_id")
            self.network_resource.append(metrics)
            extra = _container_kwargs(metrics)
            if metrics.get("errors"):
                for error_event in metrics["errors"]:
                    event = NPUHardwareEvent(hardware_type=self.name,
                                             error_code=self.error_code(error_event),
                                             node_name=node_name,
                                             card_id=card_id,
                                             device_id=device_id,
                                             logic_id=logic_id,
                                             phy_id=phy_id,
                                             fault_type=FaultType.HARDWARE,
                                             details=error_event["error_message"],
                                             **extra)
                    events.append(event)
        return events


class NPUErrorRules(ErrorRule):
    """
    NPUErrorRules detects generic NPU core errors based on error codes and reported faults.
    Also detects frequency throttling: after 4 samples where current_freq < rated_freq
    (and optionally utilization > 0), reports either high temperature or other error.
    """

    def __init__(self):
        super().__init__("NPU")
        self.npu_resource = []
        self._freq_temp_history: Dict[str, deque] = {}

    def error_code(self, event) -> str:
        return str(event["error_code"])

    def detect(self, data: dict) -> Optional[List[NPUHardwareEvent]]:
        events = []
        self.npu_resource = []
        npu_data = data.get("collector.npu.npu", {})

        # test
        # keys_without_last_two = [key for key in ERROR_CODE_MAP.keys() if key not in [0x100, 0x101]]
        # error_code_samples = random.sample(keys_without_last_two, 2)
        # error_code_samples_int = [int(code, 16) for code in error_code_samples]
        #
        # test_data = {
        #     "node_name": "test",
        #     "phy_id": 101,
        #     "device_id": 101,
        #     "card_id": 101,
        #     "namespace": "",
        #     "pod_name": "",
        #     "container_name": "",
        #     "npu_chip_info_error_code": error_code_samples_int
        #     }
        # npu_data[101] = test_data
        #
        # test_data = {
        #     "node_name": "test",
        #     "phy_id": 102,
        #     "device_id": 102,
        #     "card_id": 102,
        #     "namespace": "",
        #     "pod_name": "",
        #     "container_name": "",
        #     "npu_chip_info_aicore_utilization_rate": 0.5,
        #     "npu_chip_info_temperature": 85.0,
        #     "npu_chip_info_aicore_current_freq": 1.5,
        #     "npu_chip_info_aicore_rated_freq": 2.0
        # }
        # npu_data[102] = test_data

        for logic_id, metrics in npu_data.items():
            node_name = metrics.get("node_name")
            device_id = metrics.get("device_id")
            card_id = metrics.get("card_id")
            phy_id = metrics.get("phy_id")
            self.npu_resource.append(metrics)
            extra = _container_kwargs(metrics)
            if metrics.get("npu_chip_info_error_code"):
                for errcode in metrics["npu_chip_info_error_code"]:
                    try:
                        detail, _, err = get_device_error_code_string(logic_id, card_id, device_id,
                                                                      errcode)
                        event = NPUHardwareEvent(hardware_type=self.name,
                                                 error_code=hex(errcode),
                                                 node_name=node_name,
                                                 card_id=card_id,
                                                 device_id=device_id,
                                                 logic_id=logic_id,
                                                 phy_id=phy_id,
                                                 fault_type=FaultType.HARDWARE,
                                                 details=detail,
                                                 **extra)
                        events.append(event)
                    except Exception as e:  # pylint: disable=broad-except
                        logger.error("Failed to obtain NPU error message: %s", str(e))
            if metrics.get("errors"):
                for err in metrics["errors"]:
                    event = NPUHardwareEvent(hardware_type=self.name,
                                             error_code=self.error_code(err),
                                             node_name=node_name,
                                             card_id=card_id,
                                             device_id=device_id,
                                             logic_id=logic_id,
                                             phy_id=phy_id,
                                             fault_type=FaultType.HARDWARE,
                                             details=err["error_message"],
                                             **extra)
                    events.append(event)

            _freq_reduction_event(
                logic_id, node_name, card_id, device_id, phy_id, metrics,
                self._freq_temp_history, events,
                temp_threshold=NPU_CHIP_TEMPERATURE_THRESHOLD,
            )
        return events


def _parse_metric(value: Any) -> Optional[float]:
    """Parse a metric value; return None if missing or 'NA' or invalid."""
    if value is None or value == "NA":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _freq_reduction_event(
        logic_id: int,
        node_name: Any,
        card_id: Any,
        device_id: Any,
        phy_id: Any,
        metrics: dict,
        history: Dict[str, deque],
        events: List[NPUHardwareEvent],
        temp_threshold: float,
) -> None:
    """
    Collect one sample of utilization, temperature, current_freq, rated_freq per device.
    After 4 samples, if for all 4 points current_freq < rated_freq (and optionally
    utilization > 0), emit either a high-temperature event or an other-error event.
    """
    key = str(logic_id)
    utilization = _parse_metric(metrics.get("npu_chip_info_aicore_utilization_rate"))
    temperature = _parse_metric(metrics.get("npu_chip_info_temperature"))
    current_freq = _parse_metric(metrics.get("npu_chip_info_aicore_current_freq"))
    rated_freq = _parse_metric(metrics.get("npu_chip_info_aicore_rated_freq"))
    if utilization is None or temperature is None or current_freq is None or rated_freq is None:
        return

    point = (utilization, temperature, current_freq, rated_freq)
    if key not in history:
        history[key] = deque(maxlen=FREQ_TEMP_HISTORY_SIZE)
    history[key].append(point)
    if len(history[key]) < FREQ_TEMP_HISTORY_SIZE:
        return

    points = list(history[key])

    max_rated_freq = max(p[3] for p in points)
    freq_below_rated = all(p[2] < max_rated_freq for p in points)
    util_positive = all(p[0] > 0 for p in points)
    if not (freq_below_rated and util_positive):
        history[key].popleft()
        return

    extra = _container_kwargs(metrics)
    at_least_one_temp_above = any(p[1] > temp_threshold for p in points)
    if at_least_one_temp_above:
        detail = "NPU frequency reduction due to high temperature"
        event = NPUHardwareEvent(
            hardware_type="NPU",
            error_code=hex(100),
            node_name=node_name,
            card_id=card_id,
            device_id=device_id,
            logic_id=logic_id,
            phy_id=phy_id,
            fault_type=FaultType.HARDWARE,
            details=detail,
            **extra,
        )
        events.append(event)
    else:
        detail = (
            "NPU frequency reduction due to overload"
        )
        event = NPUHardwareEvent(
            hardware_type="NPU",
            error_code=hex(101),
            node_name=node_name,
            card_id=card_id,
            device_id=device_id,
            logic_id=logic_id,
            phy_id=phy_id,
            fault_type=FaultType.HARDWARE,
            details=detail,
            **extra,
        )
        events.append(event)
    history[key].clear()
