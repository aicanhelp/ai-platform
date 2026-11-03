# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for RDMACollector."""

import unittest

from unittest.mock import patch, mock_open, MagicMock

from network_performance_exporter.collectors.collector_for_rdma import RDMACollector


class TestGetBandwidthLimit(unittest.TestCase):
    """Tests for RDMACollector._get_bandwidth_limit."""

    def test_file_not_found_returns_none(self):
        """
        Tests RDMACollector._get_bandwidth_limit returns None when the rate file is not found.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            result = collector._get_bandwidth_limit("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertIsNone(result)

    def test_returns_mbps(self):
        """
        Tests RDMACollector._get_bandwidth_limit returns the bandwidth in MB/s when the rate file is readable.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data="100 Gb/sec (4X EDR)\n")):
            result = collector._get_bandwidth_limit("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 12500.0)  # 100 * 125


class TestGetTrafficRates(unittest.TestCase):
    """Tests for RDMACollector._get_traffic_rates."""

    def test_no_history_returns_none(self):
        """
        Tests RDMACollector._get_traffic_rates returns (None, None) when no prior traffic history exists.
        """
        collector = RDMACollector(node_name="test-node", update_interval=15)
        tx_rate, rx_rate = collector._get_traffic_rates("mlx5_0", 1, 1000.0, 2000.0)  # pylint: disable=protected-access
        self.assertIsNone(tx_rate)
        self.assertIsNone(rx_rate)
        self.assertIn("mlx5_0:1", collector._traffic_history)  # pylint: disable=protected-access

    def test_with_history_returns_rates(self):
        """
        Tests RDMACollector._get_traffic_rates returns computed MB/s rates when history is available.
        """
        collector = RDMACollector(node_name="test-node", update_interval=10)
        collector._get_traffic_rates("mlx5_0", 1, 1000.0, 2000.0)  # pylint: disable=protected-access
        # delta: tx=+100MB, rx=+200MB in 10s → 10 MB/s, 20 MB/s
        tx_rate, rx_rate = collector._get_traffic_rates(  # pylint: disable=protected-access
            "mlx5_0", 1, 1000.0 + 100 * 1024 * 1024, 2000.0 + 200 * 1024 * 1024
        )
        self.assertAlmostEqual(tx_rate, 10.0)
        self.assertAlmostEqual(rx_rate, 20.0)

    def test_none_bytes_returns_none(self):
        """
        Tests RDMACollector._get_traffic_rates returns (None, None) when tx_bytes and rx_bytes are None.
        """
        collector = RDMACollector(node_name="test-node", update_interval=15)
        tx_rate, rx_rate = collector._get_traffic_rates("mlx5_0", 1, None, None)  # pylint: disable=protected-access
        self.assertIsNone(tx_rate)
        self.assertIsNone(rx_rate)

    def test_counter_reset_clamps_to_zero(self):
        """
        Tests RDMACollector._get_traffic_rates returns 0.0 (not negative) when counters
        decrease due to a reset or overflow, and updates history with the new values.
        """
        collector = RDMACollector(node_name="test-node", update_interval=10)
        collector._get_traffic_rates("mlx5_0", 1, 1000.0 * 1024 * 1024, 2000.0 * 1024 * 1024)  # pylint: disable=protected-access
        tx_rate, rx_rate = collector._get_traffic_rates(  # pylint: disable=protected-access
            "mlx5_0", 1, 100.0 * 1024 * 1024, 200.0 * 1024 * 1024
        )
        self.assertEqual(tx_rate, 0.0)
        self.assertEqual(rx_rate, 0.0)
        history = collector._traffic_history["mlx5_0:1"]  # pylint: disable=protected-access
        self.assertEqual(history["rdma_tx_bytes_total"], 100.0 * 1024 * 1024)
        self.assertEqual(history["rdma_rx_bytes_total"], 200.0 * 1024 * 1024)

    def test_history_updated_after_call(self):
        """
        Tests RDMACollector._get_traffic_rates behavior when history is updated after successive calls.
        """
        collector = RDMACollector(node_name="test-node", update_interval=15)
        collector._get_traffic_rates("mlx5_0", 1, 1000.0, 2000.0)  # pylint: disable=protected-access
        collector._get_traffic_rates("mlx5_0", 1, 3000.0, 4000.0)  # pylint: disable=protected-access
        history = collector._traffic_history["mlx5_0:1"]  # pylint: disable=protected-access
        self.assertEqual(history["rdma_tx_bytes_total"], 3000.0)
        self.assertEqual(history["rdma_rx_bytes_total"], 4000.0)


class TestGetTrafficBytes(unittest.TestCase):
    """Tests for RDMACollector._get_traffic_bytes."""

    def test_file_not_found_returns_none(self):
        """
        Tests RDMACollector._get_traffic_bytes returns (None, None) when the counter files are not found.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            tx, rx = collector._get_traffic_bytes("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertIsNone(tx)
        self.assertIsNone(rx)

    def test_returns_bytes(self):
        """
        Tests RDMACollector._get_traffic_bytes returns scaled byte counts when counter files are readable.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", side_effect=[
            mock_open(read_data="100\n")(),
            mock_open(read_data="200\n")(),
        ]):
            tx, rx = collector._get_traffic_bytes("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertAlmostEqual(tx, 400.0)  # 100 * 4
        self.assertAlmostEqual(rx, 800.0)  # 200 * 4


class TestGetPhysicalLinkState(unittest.TestCase):
    """Tests for RDMACollector._get_physical_link_state."""

    def test_file_not_found_returns_zero(self):
        """
        Tests RDMACollector._get_physical_link_state returns 0 when the state file is not found.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            result = collector._get_physical_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 0)

    def test_physical_link_up(self):
        """
        Tests RDMACollector._get_physical_link_state returns 1 when the physical link state is LinkUp.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data="5: LinkUp\n")):
            result = collector._get_physical_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 1)

    def test_physical_link_down(self):
        """
        Tests RDMACollector._get_physical_link_state returns 0 when the physical link state is not LinkUp.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data="2: Polling\n")):
            result = collector._get_physical_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 0)


class TestGetLinkState(unittest.TestCase):
    """Tests for RDMACollector._get_link_state and _get_physical_link_state."""

    def test_file_not_found_returns_zero(self):
        """
        Tests RDMACollector._get_link_state returns 0 when the state file is not found.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            result = collector._get_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 0)

    def test_link_active(self):
        """
        Tests RDMACollector._get_link_state returns 1 when the link state is ACTIVE.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data="4: ACTIVE\n")):
            result = collector._get_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 1)

    def test_link_down(self):
        """
        Tests RDMACollector._get_link_state returns 0 when the link state is DOWN.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data="1: DOWN\n")):
            result = collector._get_link_state("mlx5_0", 1)  # pylint: disable=protected-access
        self.assertEqual(result, 0)


class TestGetRdmaDevices(unittest.TestCase):
    """Tests for RDMACollector._get_rdma_devices."""

    def test_infiniband_dir_missing_returns_empty(self):
        """
        Tests RDMACollector._get_rdma_devices returns [] when the InfiniBand directory does not exist.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("os.path.isdir", return_value=False):
            devices = collector._get_rdma_devices()  # pylint: disable=protected-access
        self.assertEqual(devices, [])

    def test_listdir_error_returns_empty(self):
        """
        Tests RDMACollector._get_rdma_devices returns [] when listing the InfiniBand directory raises an error.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("os.path.isdir", return_value=True), \
                patch("os.listdir", side_effect=OSError("permission denied")):
            devices = collector._get_rdma_devices()  # pylint: disable=protected-access
        self.assertEqual(devices, [])

    def test_returns_all_devices(self):
        """
        Tests RDMACollector._get_rdma_devices returns all devices sorted when the directory is readable.
        """
        collector = RDMACollector(node_name="test-node")
        with patch("os.path.isdir", return_value=True), \
                patch("os.listdir", return_value=["mlx5_2", "mlx5_0", "mlx5_1"]):
            devices = collector._get_rdma_devices()  # pylint: disable=protected-access
        self.assertEqual(devices, ["mlx5_0", "mlx5_1", "mlx5_2"])


class TestCollectImpl(unittest.TestCase):
    """Tests for RDMACollector._collect_impl."""

    def _make_collector(self):
        return RDMACollector(node_name="test-node", update_interval=10)

    def _mock_device_metrics(self, collector, device, port, tx_bytes, rx_bytes, link=1, phys=1, bw=12500.0):
        collector._get_link_state = MagicMock(return_value=link)  # pylint: disable=protected-access
        collector._get_physical_link_state = MagicMock(return_value=phys)  # pylint: disable=protected-access
        collector._get_traffic_bytes = MagicMock(return_value=(tx_bytes, rx_bytes))  # pylint: disable=protected-access
        collector._get_bandwidth_limit = MagicMock(return_value=bw)  # pylint: disable=protected-access
        collector._get_device_ports = MagicMock(return_value=[port])  # pylint: disable=protected-access

    def test_collect_returns_empty_on_error(self):
        """
        Tests RDMACollector._collect_impl returns {} when _get_rdma_devices raises an unexpected error.
        """
        collector = self._make_collector()
        with patch.object(collector, "_get_rdma_devices", side_effect=RuntimeError("unexpected")):
            result = collector._collect_impl()  # pylint: disable=protected-access
        self.assertEqual(result, {})

    def test_collect_removes_offline_devices_from_history(self):
        """
        Tests RDMACollector._collect_impl behavior when a device previously in history is no longer returned by _get_rdma_devices.
        """
        collector = self._make_collector()
        collector._traffic_history["ghost:1"] = {  # pylint: disable=protected-access
            "rdma_tx_bytes_total": 0.0, "rdma_rx_bytes_total": 0.0
        }
        with patch.object(collector, "_get_rdma_devices", return_value=[]):
            collector._collect_impl()  # pylint: disable=protected-access
        self.assertNotIn("ghost:1", collector._traffic_history)  # pylint: disable=protected-access

    def test_collect_port_down_omits_traffic_metrics(self):
        """
        Tests RDMACollector._collect_impl behavior when a port is down and traffic metrics are omitted from the result.
        """
        collector = self._make_collector()
        self._mock_device_metrics(collector, "mlx5_0", 1, 1000.0, 2000.0, link=0, phys=0)
        with patch.object(collector, "_get_rdma_devices", return_value=["mlx5_0"]):
            result = collector._collect_impl()  # pylint: disable=protected-access

        port_metrics = result["mlx5_0"][1]
        self.assertEqual(port_metrics["rdma_link_state"], 0)
        self.assertEqual(port_metrics["rdma_physical_link_state"], 0)
        self.assertNotIn("rdma_tx_bytes_total", port_metrics)
        self.assertNotIn("rdma_rx_bytes_total", port_metrics)
        self.assertNotIn("rdma_tx_rate_mbps", port_metrics)
        self.assertNotIn("rdma_rx_rate_mbps", port_metrics)
        # history should not be touched
        self.assertNotIn("mlx5_0:1", collector._traffic_history)  # pylint: disable=protected-access

    def test_collect_no_history_omits_rate_metrics(self):
        """
        Tests RDMACollector._collect_impl returns byte totals but omits rate metrics on the first collection call.
        """
        collector = self._make_collector()
        self._mock_device_metrics(collector, "mlx5_0", 1, 1000.0, 2000.0)
        with patch.object(collector, "_get_rdma_devices", return_value=["mlx5_0"]):
            result = collector._collect_impl()  # pylint: disable=protected-access

        port_metrics = result["mlx5_0"][1]
        self.assertEqual(port_metrics["node"], "test-node")
        self.assertEqual(port_metrics["device"], "mlx5_0")
        self.assertEqual(port_metrics["rdma_link_state"], 1)
        self.assertEqual(port_metrics["rdma_physical_link_state"], 1)
        self.assertEqual(port_metrics["rdma_tx_bytes_total"], 1000.0)
        self.assertEqual(port_metrics["rdma_rx_bytes_total"], 2000.0)
        self.assertNotIn("rdma_tx_rate_mbps", port_metrics)
        self.assertNotIn("rdma_rx_rate_mbps", port_metrics)
        self.assertNotIn("rdma_tx_available_bandwidth_mbps", port_metrics)
        self.assertNotIn("rdma_rx_available_bandwidth_mbps", port_metrics)

        history = collector._traffic_history["mlx5_0:1"]  # pylint: disable=protected-access
        self.assertEqual(history["rdma_tx_bytes_total"], 1000.0)
        self.assertEqual(history["rdma_rx_bytes_total"], 2000.0)

    def test_collect_rates_with_history(self):
        """
        Tests RDMACollector._collect_impl returns computed rate and available bandwidth metrics when history is available from a prior call.
        """
        collector = self._make_collector()
        self._mock_device_metrics(collector, "mlx5_0", 1, 1000.0, 2000.0)
        with patch.object(collector, "_get_rdma_devices", return_value=["mlx5_0"]):
            collector._collect_impl()  # pylint: disable=protected-access

        # Second collect: tx+100MB, rx+200MB in 10s → 10 MB/s, 20 MB/s
        self._mock_device_metrics(
            collector, "mlx5_0", 1,
            1000.0 + 100 * 1024 * 1024,
            2000.0 + 200 * 1024 * 1024,
        )
        with patch.object(collector, "_get_rdma_devices", return_value=["mlx5_0"]):
            result = collector._collect_impl()  # pylint: disable=protected-access

        port_metrics = result["mlx5_0"][1]
        self.assertEqual(port_metrics["rdma_tx_bytes_total"], 1000.0 + 100 * 1024 * 1024)
        self.assertEqual(port_metrics["rdma_rx_bytes_total"], 2000.0 + 200 * 1024 * 1024)
        self.assertAlmostEqual(port_metrics["rdma_tx_rate_mbps"], 10.0)
        self.assertAlmostEqual(port_metrics["rdma_rx_rate_mbps"], 20.0)
        self.assertAlmostEqual(port_metrics["rdma_tx_available_bandwidth_mbps"], 12490.0)
        self.assertAlmostEqual(port_metrics["rdma_rx_available_bandwidth_mbps"], 12480.0)

        history = collector._traffic_history["mlx5_0:1"]  # pylint: disable=protected-access
        self.assertEqual(history["rdma_tx_bytes_total"], 1000.0 + 100 * 1024 * 1024)
        self.assertEqual(history["rdma_rx_bytes_total"], 2000.0 + 200 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
