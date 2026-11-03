# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Disk metrics collector."""

import logging
import os
from typing import Any, Dict

from common.base.base_metrics_collector import BaseCollector

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class DiskStatsConstants:
    """Constants for /proc/diskstats parsing."""
    DISKSTATS_PATH = "/proc/diskstats"

    FIELD_DEVICE = 2
    FIELD_READS_COMPLETED = 3
    FIELD_SECTORS_READ = 5
    FIELD_READ_TIME_MS = 6
    FIELD_WRITES_COMPLETED = 7
    FIELD_SECTORS_WRITTEN = 9
    FIELD_WRITE_TIME_MS = 10

    MIN_FIELDS = 14

    SECTOR_SIZE = 512


class DiskCollector(BaseCollector):
    """Collector for disk metrics."""

    def __init__(self, node_name: str = "", update_interval: int = 15):
        super().__init__(update_interval=update_interval)
        self.node_name = node_name or os.getenv("NODE_NAME", "")
        self._io_history = {}

    @property
    def name(self) -> str:
        """Get collector name."""
        return "disk"

    def _collect_impl(self) -> Dict[str, Any]:
        """
        Collect disk metrics from /proc/diskstats.

        Returns:
            Dictionary with disk metrics for all devices
        """
        result = {}
        try:
            devices_data = self._parse_diskstats()
            # Clean up history for offline devices to prevent memory leaks
            # and incorrect metrics when devices come back online
            devices_to_remove = set(self._io_history.keys()) - set(devices_data.keys())
            for device in devices_to_remove:
                del self._io_history[device]

            for device, data in devices_data.items():
                metrics = {
                    "node": self.node_name,
                    "device": device,
                    "disk_reads_completed_total": data["reads_completed"],
                    "disk_read_bytes_total": data["read_sectors"] * DiskStatsConstants.SECTOR_SIZE,
                    "disk_read_time_ms_total": data["read_time_ms"],
                    "disk_writes_completed_total": data["writes_completed"],
                    "disk_write_bytes_total": data["write_sectors"] * DiskStatsConstants.SECTOR_SIZE,
                    "disk_write_time_ms_total": data["write_time_ms"],
                }

                # Calculate IOPS and throughput
                read_iops, write_iops, read_throughput_mb, write_throughput_mb = self._calculate_iops_and_throughput(
                    device,
                    data["reads_completed"],
                    data["writes_completed"],
                    data["read_sectors"],
                    data["write_sectors"],
                )
                if read_iops is not None:
                    metrics["disk_read_iops"] = read_iops
                if write_iops is not None:
                    metrics["disk_write_iops"] = write_iops
                if read_throughput_mb is not None:
                    metrics["disk_read_throughput_mb"] = read_throughput_mb
                if write_throughput_mb is not None:
                    metrics["disk_write_throughput_mb"] = write_throughput_mb

                # Calculate utilization
                read_utilization, write_utilization = self._calculate_utilization(
                    device, data["read_time_ms"], data["write_time_ms"]
                )
                if read_utilization is not None:
                    metrics["disk_read_utilization"] = read_utilization
                if write_utilization is not None:
                    metrics["disk_write_utilization"] = write_utilization

                result[device] = metrics
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to collect disk metrics: %s", str(e))

        return result

    def _calculate_utilization(self, device: str, read_time_ms: int, write_time_ms: int) -> tuple:
        """
        Calculate disk utilization based on I/O time.

        Args:
            device: Device name
            read_time_ms: Total time spent reading (milliseconds)
            write_time_ms: Total time spent writing (milliseconds)

        Returns:
            Tuple of (read_utilization, write_utilization) as percentages
        """
        if "read_time_ms" not in self._io_history.get(device, {}):
            self._io_history[device]["read_time_ms"] = read_time_ms
            self._io_history[device]["write_time_ms"] = write_time_ms
            return None, None

        history = self._io_history[device]
        time_delta_ms = self.update_interval * 1000

        read_utilization = None
        write_utilization = None

        if time_delta_ms > 0:
            read_time_delta = read_time_ms - history.get("read_time_ms")
            write_time_delta = write_time_ms - history.get("write_time_ms")

            if read_time_delta >= 0:
                read_utilization = (read_time_delta / time_delta_ms) * 100
            if write_time_delta >= 0:
                write_utilization = (write_time_delta / time_delta_ms) * 100

        # Update history with time fields
        self._io_history[device]["read_time_ms"] = read_time_ms
        self._io_history[device]["write_time_ms"] = write_time_ms

        return read_utilization, write_utilization

    def _calculate_iops_and_throughput(
        self, device: str, reads_completed: int, writes_completed: int, read_sectors: int, write_sectors: int
    ) -> tuple:
        """
        Calculate IOPS and throughput based on history.

        Args:
            device: Device name
            reads_completed: Total reads completed
            writes_completed: Total writes completed
            read_sectors: Total sectors read
            write_sectors: Total sectors written

        Returns:
            Tuple of (read_iops, write_iops, read_throughput_mb, write_throughput_mb),
            None for each value that cannot be calculated.
        """
        if device not in self._io_history:
            self._io_history[device] = {
                "reads_completed": reads_completed,
                "writes_completed": writes_completed,
                "read_sectors": read_sectors,
                "write_sectors": write_sectors,
            }
            return None, None, None, None

        history = self._io_history[device]
        time_delta = self.update_interval

        read_iops = None
        write_iops = None
        read_throughput_mb = None
        write_throughput_mb = None

        if time_delta > 0:
            read_delta = reads_completed - history["reads_completed"]
            write_delta = writes_completed - history["writes_completed"]
            read_sectors_delta = read_sectors - history["read_sectors"]
            write_sectors_delta = write_sectors - history["write_sectors"]

            if read_delta >= 0:
                read_iops = read_delta / time_delta
            if write_delta >= 0:
                write_iops = write_delta / time_delta
            if read_sectors_delta >= 0:
                read_throughput_mb = (read_sectors_delta * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * time_delta)
            if write_sectors_delta >= 0:
                write_throughput_mb = (write_sectors_delta * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * time_delta)

        # Update history
        self._io_history[device].update({
            "reads_completed": reads_completed,
            "writes_completed": writes_completed,
            "read_sectors": read_sectors,
            "write_sectors": write_sectors,
        })

        return read_iops, write_iops, read_throughput_mb, write_throughput_mb

    def _parse_diskstats(self) -> Dict[str, Dict[str, int]]:
        """
        Parse /proc/diskstats once and return all devices' raw data.

        Returns:
            Dictionary with device names as keys and their stats as values
        """
        devices_data = {}
        try:
            with open(DiskStatsConstants.DISKSTATS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= DiskStatsConstants.MIN_FIELDS:
                        device = parts[DiskStatsConstants.FIELD_DEVICE]
                        devices_data[device] = {
                            "reads_completed": int(parts[DiskStatsConstants.FIELD_READS_COMPLETED]),
                            "read_sectors": int(parts[DiskStatsConstants.FIELD_SECTORS_READ]),
                            "read_time_ms": int(parts[DiskStatsConstants.FIELD_READ_TIME_MS]),
                            "writes_completed": int(parts[DiskStatsConstants.FIELD_WRITES_COMPLETED]),
                            "write_sectors": int(parts[DiskStatsConstants.FIELD_SECTORS_WRITTEN]),
                            "write_time_ms": int(parts[DiskStatsConstants.FIELD_WRITE_TIME_MS]),
                        }
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to read %s: %s", DiskStatsConstants.DISKSTATS_PATH, str(e))

        return devices_data
