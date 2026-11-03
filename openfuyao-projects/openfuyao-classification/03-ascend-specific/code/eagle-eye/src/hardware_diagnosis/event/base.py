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
This module defines FaultType, Event classes.
"""

import uuid
import datetime

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class FaultType(Enum):
    """
    Fault Type enum.
    """

    HARDWARE = "collector"
    SOFTWARE = "software"


@dataclass
class Event:
    """
    Base class for all type events.
    """
    def __init__(self, node_name: str,
                 fault_type: FaultType, details: str,
                 timestamp: Optional[str] = None,
                 severity: int = 1,
                 event_id: Optional[str] = None):
        self.event_id = event_id or str(uuid.uuid4())
        self.node_name = node_name
        self.timestamp = timestamp or str(datetime.datetime.now())
        self.fault_type = fault_type
        self.details = details
        self.severity = severity
