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
This module defines the HardwareEvent class.
"""

from typing import Optional

from hardware_diagnosis.event.base import Event, FaultType


class NPUHardwareEvent(Event):
    """
    Describes a NPU Hardware Event. Includes device identifiers and optional
    container context (namespace, pod_name, container_name) from hardware monitor.
    """
    def __init__(self, hardware_type: str, error_code: str,
                 device_id: int, logic_id: int, card_id: int, phy_id: int,
                 node_name: str, fault_type: FaultType, details: str, timestamp: Optional[str] = None,
                 severity: int = 1, event_id: Optional[str] = None,
                 namespace: Optional[str] = None, pod_name: Optional[str] = None, container_name: Optional[str] = None):
        super().__init__(node_name, fault_type, details, timestamp, severity, event_id)
        self.device_id = device_id
        self.logic_id = logic_id
        self.card_id = card_id
        self.phy_id = phy_id
        self.error_code = error_code
        self.hardware_type = hardware_type
        self.namespace = namespace or ""
        self.pod_name = pod_name or ""
        self.container_name = container_name or ""
        self.id_info = {
            "device_id": self.device_id,
            "logic_id": self.logic_id,
            "card_id": self.card_id,
            "phy_id": self.phy_id,
            "namespace": self.namespace,
            "pod_name": self.pod_name,
            "container_name": self.container_name,
        }

    def __str__(self) -> str:
        return (
            f"NPUHardwareEvent(error_code={self.error_code!r}, node_name={self.node_name!r}, "
            f"device_id={self.device_id!r}, card_id={self.card_id!r}, logic_id={self.logic_id!r}, "
            f"phy_id={self.phy_id!r}, namespace={self.namespace!r}, pod_name={self.pod_name!r}, "
            f"container_name={self.container_name!r}, details={self.details!r}, timestamp={self.timestamp!r}, "
            f"fault_type={self.fault_type!r}, hardware_type={self.hardware_type!r})")

    def __repr__(self) -> str:
        return (
            f"NPUHardwareEvent(error_code={self.error_code!r}, node_name={self.node_name!r}, "
            f"device_id={self.device_id!r}, card_id={self.card_id!r}, logic_id={self.logic_id!r}, "
            f"phy_id={self.phy_id!r}, namespace={self.namespace!r}, pod_name={self.pod_name!r}, "
            f"container_name={self.container_name!r}, details={self.details!r}, timestamp={self.timestamp!r}, "
            f"fault_type={self.fault_type!r}, hardware_type={self.hardware_type!r})")
