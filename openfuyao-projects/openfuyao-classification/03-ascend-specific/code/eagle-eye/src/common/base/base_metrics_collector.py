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
This module defines BaseCollector class.
"""

import os
import logging
import threading
import time

from abc import ABC, abstractmethod
from typing import Any
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from common.base.cache_manager import CacheManager

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class BaseCollector(ABC):
    """
    BaseCollector is the base class for collector collectors and software collectors.
    """

    executor = ThreadPoolExecutor(max_workers=30)

    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._collect_task, daemon=True)
        self.result_queue = Queue()
        self._start_barrier = None

        self.cache = CacheManager()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property to get the name of the collector.
        """
        pass

    def start(self):
        """
        Starts the collector's data collection thread.
        This will begin the collection process in the background.
        """
        self._thread.start()

    def set_start_barrier(self, barrier):
        """
        Set the barrier for synchronizing startup with other collectors.

        Args:
            barrier: threading.Barrier instance for synchronization
        """
        self._start_barrier = barrier

    def stop(self):
        """
        Stops the collector's data collection thread.
        This will stop the collection process and wait for the thread to finish.
        """
        self._stop_event.set()
        self._thread.join()

    def _collect_task(self):
        """
        The task that runs in the background to collect data at the specified interval.
        This function runs continuously until the stop event is set, calling the collect method.
        """
        # Wait for all collectors to be ready (if barrier is set)
        if self._start_barrier:
            self._start_barrier.wait()

        while not self._stop_event.is_set():
            start_time = time.time()
            self.collect()
            elapsed = time.time() - start_time
            remaining = self.update_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def collect(self) -> Any:
        """
        Performs data collection and stores the result in the queue.
        Each round of collection produces one data item that is put into the queue.
        The queue is never cleared, ensuring no data is lost.
        """
        future = self.executor.submit(self._collect_impl)

        try:
            result = future.result(timeout=self.update_interval)
            self.result_queue.put(result)
        except TimeoutError:
            logger.warning("Timeout occurred while collecting data for %s.", self.name)
            self.result_queue.put({})
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while collecting data for %s: %s", self.name, str(e))
            self.result_queue.put({})

    @abstractmethod
    def _collect_impl(self) -> Any:
        """
        Abstract method to be implemented by subclasses for the actual data collection logic.
        """
        pass
