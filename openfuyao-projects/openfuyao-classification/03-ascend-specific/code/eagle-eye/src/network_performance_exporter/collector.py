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
Unified network performance metrics collector.
"""

import asyncio
import os
import time
import queue
import threading

from typing import Any, Dict

from common.npu_device_info.collector_for_npu_device_info import NPUDeviceInfoCollector
from logger.setup_logger import setup_logger
from network_performance_exporter.collectors.collector_for_rdma import RDMACollector
from network_performance_exporter.collectors.collector_for_npu_roce import NPURoCECollector
from network_performance_exporter.collectors.collector_for_disk import DiskCollector
from network_performance_exporter.config import Config
from network_performance_exporter.exporters.prometheus_exporter import PrometheusExporter
from network_performance_exporter.exporters.nats_exporter import NatsExporter
from utils.utils import detect_devices

module_name = os.getenv("MODULE_NAME", "default")
logger = setup_logger()


class NetworkPerformanceCollector:
    """Unified collector for network performance metrics (coordinator)."""

    def __init__(self, config: Config, prometheus_exporter: PrometheusExporter,
                 nats_exporter: NatsExporter, update_interval: int = 15):
        self.config = config
        self.prometheus_exporter = prometheus_exporter
        self.nats_exporter = nats_exporter
        self.update_interval = update_interval
        self._started_collectors = []
        self._start_barrier = None

    def bootstrap(self) -> None:
        """初始化并启动所有必要的 Collector（根据可用设备）。"""
        devices = detect_devices()
        has_npu = devices.get("npu", False)
        logger.info("Device detection result: has_npu=%s", has_npu)

        if has_npu:
            self._bootstrap_npu_collectors()
        else:
            logger.info("NPU not detected, skipping NPU RoCE Collector")

        self._create_rdma_and_disk_collectors()
        self._setup_barrier()

    def _bootstrap_npu_collectors(self) -> None:
        """Bootstrap NPU-related collectors."""
        logger.info("NPU detected, creating NPU RoCE Collector...")
        npu_roce_collector = NPURoCECollector(self.config.node_name, self.update_interval)
        self._started_collectors.append(("npu_roce", npu_roce_collector))

        npu_device_collector = NPUDeviceInfoCollector()
        try:
            npu_device_collector.collect_info()
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Initial NPU device collection failed (thread will retry): %s", str(e))
        npu_device_collector.start()

    def _create_rdma_and_disk_collectors(self) -> None:
        """Create RDMA and Disk collectors."""
        rdma_collector = RDMACollector(self.config.node_name, self.update_interval)
        disk_collector = DiskCollector(self.config.node_name, self.update_interval)
        self._started_collectors.append(("rdma", rdma_collector))
        self._started_collectors.append(("disk", disk_collector))
        logger.info("All collectors created: %d collectors", len(self._started_collectors))

    def _setup_barrier(self) -> None:
        barrier = threading.Barrier(len(self._started_collectors) + 1)
        self._start_barrier = barrier
        for _, collector in self._started_collectors:
            collector.set_start_barrier(barrier)
        logger.info("Barrier created with %d participants", len(self._started_collectors) + 1)

    def start(self):
        """Start all sub-collectors simultaneously."""
        for name, collector in self._started_collectors:
            collector.start()
            logger.info("Started collector: %s", name)

    def stop(self):
        """Stop all sub-collectors."""
        for name, collector in self._started_collectors:
            collector.stop()
            logger.info("Stopped collector: %s", name)

    async def collect_device_data(self, collector_name: str, collector) -> Dict[str, Any]:
        """
        Collect data asynchronously from a specific collector.
        """
        try:
            result = await asyncio.to_thread(collector.result_queue.get,
                                             timeout=self.update_interval)
            logger.debug("Collected data for %s: %s", collector_name, result)
            return result
        except queue.Empty:
            logger.warning("Timeout occurred while collecting data for %s", collector_name)
            return {}
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Collector %s encountered an error: %s", collector_name, str(e))
            return {}

    async def collect_one_round(self) -> Dict[str, Any]:
        """Collect data from all collectors in parallel."""
        tasks = [self.collect_device_data(name, collector)
                 for name, collector in self._started_collectors]
        results = await asyncio.gather(*tasks)

        return {name: result for (name, _), result in zip(self._started_collectors, results)}

    async def run_forever(self, stop_event: asyncio.Event = None):
        """
        Continuous collection and publishing loop.
        Synchronizes with sub-collectors at startup using barrier.

        Args:
            stop_event: Optional asyncio.Event to signal when to stop
        """
        # Synchronize with sub-collectors at startup
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._start_barrier.wait)
        logger.info("Main program synchronized with collectors, starting collection loop")

        while True:
            # Check if stop signal received
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, exiting collection loop")
                break

            try:
                start_time = time.time()

                # Collect from all collectors in parallel
                metrics_data = await self.collect_one_round()

                # Update Prometheus exporter
                try:
                    self.prometheus_exporter.update(metrics_data)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Failed to update Prometheus exporter: %s", str(e))

                # Publish to NATS exporter
                if self.nats_exporter.is_connected():
                    try:
                        await self.nats_exporter.publish(metrics_data)
                    except Exception as e:  # pylint: disable=broad-except
                        logger.error("Failed to publish to NATS: %s", str(e))

                # Calculate remaining time to maintain stable interval
                elapsed = time.time() - start_time
                remaining = self.update_interval - elapsed

                if remaining > 0:
                    await asyncio.sleep(remaining)

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error in collection loop: %s", str(e))
                await asyncio.sleep(self.update_interval)
