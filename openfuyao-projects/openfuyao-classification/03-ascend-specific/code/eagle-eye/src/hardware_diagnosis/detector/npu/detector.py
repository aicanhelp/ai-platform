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
This module defines NPUHardwareDetector, NpuDetector classes and function obtain_npu_id.
"""

import threading
import logging
import os

from abc import ABC
from typing import List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from hardware_diagnosis.detector.npu.rules import HBMErrorRule, DDRErrorRule, NPUErrorRules, NetworkErrorRule
from hardware_diagnosis.detector.base import Detector, ErrorRule
from hardware_diagnosis.event.hardware_event import NPUHardwareEvent, FaultType

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class NPUHardwareDetector(Detector, ABC):
    """
    This class defines the function of detect, _register_rules and _create_unavailable_event.
    """

    def __init__(self, hardware_type: str):
        super().__init__()
        self.hardware_type = hardware_type
        self.rules: List[ErrorRule] = []
        self.event_callback: Optional[Callable[[List[NPUHardwareEvent]], None]] = None
        self.running = False
        self.event_lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def _register_rules(self, rules: List[ErrorRule]):
        self.rules = rules

    def detect(self, data: dict) -> dict[str, List[NPUHardwareEvent]]:
        events = {rule.name: [] for rule in self.rules}
        events["error"] = []

        futures = []
        for rule in self.rules:
            future = self.thread_pool.submit(rule.detect, data)
            futures.append((future, rule))

        for future, rule in futures:
            try:
                rule_events = future.result()
                with self.event_lock:
                    if rule_events:
                        events[rule.name] = rule_events

            except Exception as e:  # pylint: disable=broad-except
                error_event = self._create_unavailable_event({
                    "message": f"{rule.name} detect failed: {str(e)}"
                })
                with self.event_lock:
                    events["error"].append(error_event)
                logger.error("Rule %s: detection failed: %s", rule.name, e, exc_info=True)

        return events

    def _create_unavailable_event(self, error_info: dict) -> NPUHardwareEvent:
        return NPUHardwareEvent(
            hardware_type="system",
            error_code="SYSTEM_001",
            device_id=-1,
            logic_id=-1,
            card_id=-1,
            phy_id=-1,
            node_name="",
            fault_type=FaultType.SOFTWARE,
            details=error_info["message"],
            severity=3,
            namespace="",
            pod_name="",
            container_name="",
        )


class NpuDetector(NPUHardwareDetector):
    """
    This class defines the event callback function.
    """

    def __init__(self):

        super().__init__(hardware_type="npu")
        self.filter_events = None
        self._register_rules([
            HBMErrorRule(),
            DDRErrorRule(),
            NPUErrorRules(),
            NetworkErrorRule()
        ])

    def process_events(self, events: dict[str, List[NPUHardwareEvent]]):
        """
        Processes the events detected by the rules and forwards them for further handling.
        """
        self._on_events_detected(events)

    def _on_events_detected(self, events: dict[str, List[NPUHardwareEvent]]):
        """
        Filters and stores the detected hardware events for further processing or
        reporting.
        """
        filtered_events = []
        for _, event in events.items():
            for e in event:
                filtered_events.append(e)
        self.filter_events = filtered_events
        logger.info("Detected events: %s", self.filter_events)
