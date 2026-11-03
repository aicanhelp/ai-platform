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
This module defines the MainCollector class for managing device data collection and integration
with NPU/GPU monitoring.
It provides methods for initializing, collecting data, and publishing results via NATS.
"""

import threading
import asyncio
import queue
import time
from typing import Optional, Dict, Any

from common.npu_device_info.collector_for_npu_device_info import NPUDeviceInfoCollector
from hardware_monitor.base.registry import registry
from hardware_monitor.base.loader import auto_import_all_submodules
from hardware_monitor.collector.metrics import npu
from hardware_monitor.collector.container_info.containerd import ContainerInfoCollector
from logger.setup_logger import setup_logger
from nats_communicator.publisher import NatsPublisher
from utils.utils import detect_devices

logger = setup_logger()


def nested_set(dic, keys, value):
    """
    Sets a value in a nested dictionary, creating intermediate dictionaries if necessary.
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


class MainCollector:
    """
    MainCollector is responsible for managing the collection of data from various hardware devices
    (NPU/GPU). It initializes device collectors, orchestrates data collection,
    and publishes the results to a NATS server.
    """

    def __init__(
            self,
            options: Optional[Dict[str, Any]] = None,
            update_interval: int = 30
    ):
        self.options = options or {}
        self.update_interval = update_interval
        self._started_collectors = []
        self._collecting = threading.Event()
        self.nats_publisher = NatsPublisher()
        self.collected_data = {}
        self._start_barrier = None  # Barrier for synchronizing with main program

    def bootstrap(self) -> None:
        """
        Initializes and starts all necessary collectors for the available devices (NPU and GPU).
        """
        try:
            devices = detect_devices()

            if devices["npu"] or devices["gpu"]:
                if devices["npu"]:
                    logger.info("NPU detected.")

                    npu_device_collector = NPUDeviceInfoCollector()
                    try:
                        npu_device_collector.collect_info()
                    except Exception as e:  # pylint: disable=broad-except
                        logger.debug("Initial NPU device collection failed (thread will retry): %s", str(e))
                    npu_device_collector.start()

                    container_info_collector = ContainerInfoCollector()
                    try:
                        container_info_collector.collect_info()
                    except Exception as e:  # pylint: disable=broad-except
                        logger.debug("Initial container info collection failed (thread will retry): %s", str(e))
                    container_info_collector.start()

                    logger.info("Device info and container info collectors started.")

                if devices["gpu"]:
                    logger.info("GPU detected.")
            else:
                logger.info("No NPU or GPU found, skipping device collection.")
                return

            auto_import_all_submodules(npu)

            # Collect all collectors first
            all_collectors = []
            for group, collectors in registry.get_all().items():
                for name, collector in collectors.items():
                    all_collectors.append((group, name, collector))

            # Create barrier to synchronize all collectors + main program
            if all_collectors:
                # barrier includes all collectors + main program (1 extra)
                barrier = threading.Barrier(len(all_collectors) + 1)
                self._start_barrier = barrier

                # Start all collectors with barrier
                self._started_collectors = []
                for group, name, collector in all_collectors:
                    collector.update_interval = self.update_interval
                    collector.set_start_barrier(barrier)
                    collector.start()
                    self._started_collectors.append((group, name, collector))
                    logger.info("Started collector: %s.%s", group, name)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in bootstrap: %s", str(e))

    async def collect_device_data(self, collector, group, name, results):
        """
        Collects data asynchronously for a specific device from a given collector.
        """
        try:
            result = await asyncio.to_thread(collector.result_queue.get,
                                             timeout=self.update_interval)
            logger.info("Collected data for %s.%s: %s", group, name, result)
            results[f"{group}.{name}"] = result
        except queue.Empty:
            logger.warning("Timeout occurred while collecting data for %s.%s", group, name)
            results[f"{group}.{name}"] = {}
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Collector %s.%s encountered an error: %s", group, name, str(e))
            results[f"{group}.{name}"] = {}

    async def collect_one_round(self) -> dict:
        """
        Collects data from all active collectors in one round asynchronously.
        """
        results = {}

        tasks = []
        for group, name, collector in self._started_collectors:
            tasks.append(self.collect_device_data(collector, group, name, results))

        await asyncio.gather(*tasks)

        return results

    async def run_forever(self):
        """
        Starts the continuous collection and publishing loop. It collects data, publishes it to NATS,
        and repeats the process indefinitely with the specified interval.
        Synchronizes with sub-collectors at startup using barrier.
        """
        # Synchronize with sub-collectors at startup
        if self._start_barrier:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._start_barrier.wait)
            logger.info("Main program synchronized with collectors, starting collection loop")

        while True:
            if not self._started_collectors:
                logger.info("No devices detected. Skipping data collection.")
                await asyncio.sleep(self.update_interval)
                continue

            start_time = time.time()

            results = await self.collect_one_round()

            await self.nats_publisher.publish(results)

            logger.info("Collected results: %s", results)

            # Calculate remaining time to maintain stable interval
            elapsed = time.time() - start_time
            remaining = self.update_interval - elapsed

            if remaining > 0:
                await asyncio.sleep(remaining)


if __name__ == "__main__":
    async def main():
        """
        The main entry point for initializing and running the MainCollector.
        """
        main_collector = MainCollector()
        try:
            await main_collector.nats_publisher.connect()
            logger.info("Connected to NATS server successfully.")

            main_collector.bootstrap()
            logger.info("Bootstrap completed.")

            await main_collector.run_forever()

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in main execution: %s", str(e))


    asyncio.run(main())
