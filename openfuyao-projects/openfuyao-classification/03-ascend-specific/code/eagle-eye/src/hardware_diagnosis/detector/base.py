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
This module defines ErrorRule and Detector classes.
"""

import logging
import os

from abc import ABC, abstractmethod
from typing import List, Optional

from hardware_diagnosis.event.base import Event

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class ErrorRule(ABC):
    """
    Base class for defining error detection rules.
    Each type of error corresponds to a subclass that implements the specific detection logic.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def error_code(self, event: dict) -> str:
        """Return error code"""
        pass

    @abstractmethod
    def detect(self, data: dict) -> Optional[List[Event]]:
        """Return detect event"""
        pass


class Detector(ABC):
    """
    Detector is the base class for all detectors.
    """

    def __init__(self):
        self.event_callback = None

    @abstractmethod
    def detect(self, data: dict) -> dict[str, List[Event]]:
        """Detect from different rules"""
        pass

    def set_event_callback(self, callback):
        """Set callback function"""
        self.event_callback = callback

    def start(self, data: dict):
        """
        Starts the detection process on the provided data.
        """
        self._run(data)

    def _run(self, data: dict):
        try:
            events = self.detect(data)
            if events and self.event_callback:
                self.event_callback(events)

        except BaseException as e:  # pylint: disable=broad-except
            logger.error("Detector error: %s", str(e))
