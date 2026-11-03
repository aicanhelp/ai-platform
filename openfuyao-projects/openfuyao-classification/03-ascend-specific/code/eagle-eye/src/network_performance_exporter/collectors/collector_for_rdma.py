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
RDMA network card metrics collector.
"""

import logging
import os
import re

from typing import Any, Dict, List, Tuple, Optional

from common.base.base_metrics_collector import BaseCollector

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class RDMACollector(BaseCollector):
    """Collector for RDMA network card metrics."""

    BANDWIDTH_RATE_PATTERN = re.compile(r"(\d+)\s+Gb/sec")

    def __init__(self, node_name: str = "", update_interval: int = 15):
        super().__init__(update_interval=update_interval)
        self.node_name = node_name or os.getenv("NODE_NAME", "")
        self._traffic_history = {}

    @property
    def name(self) -> str:
        """Get collector name."""
        return "rdma"

    def _collect_impl(self) -> Dict[str, Any]:
        """
        Collect RDMA network card metrics.

        Returns:
            Dictionary with RDMA metrics keyed by device name
        """
        result = {}
        try:
            devices = self._get_rdma_devices()
            # Clean up history for offline devices to prevent memory leaks
            # and incorrect metrics when devices come back online
            keys_to_remove = [k for k in self._traffic_history if k.split(":")[0] not in devices]
            for key in keys_to_remove:
                del self._traffic_history[key]

            for device in devices:
                metrics = self._collect_device_metrics(device)
                if metrics:
                    result[device] = metrics
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to collect RDMA metrics: %s", str(e))

        return result

    def _get_bandwidth_limit(self, device: str, port: int) -> Optional[float]:
        """
        Get theoretical bandwidth limit in MB/s.

        Args:
            device: Device name
            port: Port number

        Returns:
            Bandwidth in MB/s, None if error
        """
        try:
            rate_file = f"/sys/class/infiniband/{device}/ports/{port}/rate"
            with open(rate_file, "r", encoding="utf-8") as f:
                rate_str = f.read().strip()
                # Rate format: "100 Gb/sec" or "200 Gb/sec"
                match = self.BANDWIDTH_RATE_PATTERN.search(rate_str)
                if match:
                    gbps = int(match.group(1))
                    # Convert Gb/s to MB/s (1 Gb/s = 125 MB/s)
                    return gbps * 125
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to get bandwidth limit for %s port %d: %s", device, port, str(e))

        return None

    def _get_traffic_rates(self, device: str, port: int, tx_bytes: Optional[float], rx_bytes: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate current traffic rates in MB/s based on two samples.

        Args:
            device: Device name
            port: Port number
            tx_bytes: Current transmitted bytes
            rx_bytes: Current received bytes

        Returns:
            Tuple of (tx_rate_mbps, rx_rate_mbps)
        """
        if tx_bytes is None or rx_bytes is None:
            return None, None

        cache_key = f"{device}:{port}"
        time_delta = self.update_interval

        # If no previous data, cache and return None
        if cache_key not in self._traffic_history:
            self._traffic_history[cache_key] = {'rdma_tx_bytes_total': tx_bytes, 'rdma_rx_bytes_total': rx_bytes}
            return None, None

        # Calculate delta
        prev = self._traffic_history[cache_key]

        # Calculate rates (bytes/sec → MB/sec)
        tx_rate = (tx_bytes - prev['rdma_tx_bytes_total']) / time_delta / (1024 * 1024)
        rx_rate = (rx_bytes - prev['rdma_rx_bytes_total']) / time_delta / (1024 * 1024)

        # Ensure rates are non-negative (handle counter resets)
        tx_rate = max(0.0, tx_rate)
        rx_rate = max(0.0, rx_rate)

        # Update cache
        self._traffic_history[cache_key] = {'rdma_tx_bytes_total': tx_bytes, 'rdma_rx_bytes_total': rx_bytes}

        return tx_rate, rx_rate

    def _get_traffic_bytes(self, device: str, port: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get cumulative traffic data in bytes.

        Args:
            device: Device name
            port: Port number

        Returns:
            Tuple of (tx_bytes, rx_bytes), (None, None) if error
        """
        try:
            tx_file = f"/sys/class/infiniband/{device}/ports/{port}/counters/port_xmit_data"
            rx_file = f"/sys/class/infiniband/{device}/ports/{port}/counters/port_rcv_data"

            with open(tx_file, "r", encoding="utf-8") as f1, open(rx_file, "r", encoding="utf-8") as f2:
                tx_bytes = float(f1.read().strip()) * 4  # Each unit is 4 bytes
                rx_bytes = float(f2.read().strip()) * 4  # Each unit is 4 bytes

            return tx_bytes, rx_bytes
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to get traffic bytes for %s port %d: %s", device, port, str(e))
            return None, None

    def _read_state_file(self, device: str, port: int, filename: str, keyword: str) -> int:
        """
        Read state file and check if keyword is present.

        Args:
            device: Device name
            port: Port number
            filename: State file name (e.g., "state", "phys_state")
            keyword: Keyword to search for (e.g., "ACTIVE", "LinkUp")

        Returns:
            1 if keyword found, 0 otherwise
        """
        try:
            file_path = f"/sys/class/infiniband/{device}/ports/{port}/{filename}"
            with open(file_path, "r", encoding="utf-8") as f:
                state = f.read().strip()
                return 1 if keyword in state else 0
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to read %s for %s port %d: %s", filename, device, port, str(e))
            return 0

    def _get_physical_link_state(self, device: str, port: int) -> int:
        """
        Get physical link state (1=LinkUp, 0=LinkDown).

        Args:
            device: Device name
            port: Port number

        Returns:
            1 if LinkUp, 0 if LinkDown
        """
        return self._read_state_file(device, port, "phys_state", "LinkUp")

    def _get_link_state(self, device: str, port: int) -> int:
        """
        Get protocol layer link state (1=ACTIVE, 0=DOWN).

        Args:
            device: Device name
            port: Port number

        Returns:
            1 if ACTIVE, 0 if DOWN
        """
        return self._read_state_file(device, port, "state", "ACTIVE")

    def _get_device_ports(self, device: str) -> List[int]:
        """
        Get all available port numbers for a device.

        Args:
            device: Device name

        Returns:
            List of port numbers (sorted)
        """
        try:
            ports_dir = f"/sys/class/infiniband/{device}/ports"
            if os.path.isdir(ports_dir):
                ports = [int(d) for d in os.listdir(ports_dir) if d.isdigit()]
                return sorted(ports)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to get ports for device %s: %s", device, str(e))
        return []

    def _collect_device_metrics(self, device: str) -> Dict[int, Dict[str, Any]]:
        """
        Collect metrics for all ports of a specific RDMA device.

        Args:
            device: Device name (e.g., "mlx5_0")

        Returns:
            Dictionary with port metrics keyed by port number
        """
        result = {}
        ports = self._get_device_ports(device)

        for port in ports:
            metrics = {
                "node": self.node_name,
                "device": device,
                "port": port,
            }

            # Get protocol layer link state
            link_state = self._get_link_state(device, port)
            metrics["rdma_link_state"] = link_state

            # Get physical link state
            phys_state = self._get_physical_link_state(device, port)
            metrics["rdma_physical_link_state"] = phys_state

            if phys_state and link_state:
                # Get traffic bytes
                tx_bytes, rx_bytes = self._get_traffic_bytes(device, port)
                if tx_bytes is not None:
                    metrics["rdma_tx_bytes_total"] = tx_bytes
                if rx_bytes is not None:
                    metrics["rdma_rx_bytes_total"] = rx_bytes

                # Get traffic rates (requires two samples)
                tx_rate, rx_rate = self._get_traffic_rates(device, port, tx_bytes, rx_bytes)
                if tx_rate is not None:
                    metrics["rdma_tx_rate_mbps"] = tx_rate
                if rx_rate is not None:
                    metrics["rdma_rx_rate_mbps"] = rx_rate

                # Get bandwidth limit
                bandwidth_limit = self._get_bandwidth_limit(device, port)

                # Calculate available bandwidth
                if bandwidth_limit is not None and tx_rate is not None:
                    metrics["rdma_tx_available_bandwidth_mbps"] = max(0.0, bandwidth_limit - tx_rate)
                if bandwidth_limit is not None and rx_rate is not None:
                    metrics["rdma_rx_available_bandwidth_mbps"] = max(0.0, bandwidth_limit - rx_rate)

            result[port] = metrics

        return result

    def _get_rdma_devices(self) -> List[str]:
        """
        Get list of RDMA devices that have at least one ACTIVE port.

        Returns:
            List of RDMA device names
        """
        devices = []
        try:
            infiniband_dir = "/sys/class/infiniband"
            if not os.path.isdir(infiniband_dir):
                return devices
            for device in sorted(os.listdir(infiniband_dir)):
                devices.append(device)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to get RDMA devices: %s", str(e))
        return devices
