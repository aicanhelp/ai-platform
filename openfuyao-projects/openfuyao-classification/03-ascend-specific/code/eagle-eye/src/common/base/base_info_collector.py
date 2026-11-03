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
This module defines the `BaseInfoCollector` class,
which serves as the base class for collecting information in a multi-threaded environment.
"""

import threading
import time

from abc import ABC, abstractmethod
from typing import Any, Dict

from common.base.cache_manager import CacheManager


class BaseInfoCollector(ABC):
    """
    A base class for information collectors,
    handling the periodic collection of device data and updating the cache.
    """
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.cache = CacheManager()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_task, daemon=True)

    def _run_task(self):
        while not self._stop_event.is_set():
            self.collect_info()
            time.sleep(self.update_interval)

    def start(self):
        """
        Starts the background thread to collect device information periodically.
        """
        self._thread.start()

    def stop(self):
        """
        Stops the background thread collecting device information.
        """
        self._stop_event.set()
        self._thread.join()

    @abstractmethod
    def collect_info(self):
        """
        Abstract method for collecting information.
        """
        pass

    def update_cache(self, device_info: Dict[str, Any]):
        """
        Updates the cache with the collected device information.
        """
        for key, value in device_info.items():
            self.cache.set(key, value)
