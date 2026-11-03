# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for DiskCollector."""

import unittest

from unittest.mock import patch, mock_open

from network_performance_exporter.collectors.collector_for_disk import DiskCollector, DiskStatsConstants

SAMPLE_DISKSTATS = (
    "   8       0 sda 100 0 2000 500 200 0 4000 1000 0 800 1500 0 0 0 0\n"
    "   8       1 sda1 50 0 1000 250 100 0 2000 500 0 400 750 0 0 0 0\n"
    "   7       0 loop0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
)


class TestParseDiskstats(unittest.TestCase):
    """Tests for DiskCollector._parse_diskstats."""

    def test_parse_file_not_found(self):
        """
        Tests DiskCollector._parse_diskstats returns {} when the diskstats file is not found.
        """
        collector = DiskCollector(node_name="test-node")
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            result = collector._parse_diskstats()  # pylint: disable=protected-access
        self.assertEqual(result, {})

    def test_parse_skips_short_lines(self):
        """
        Tests DiskCollector._parse_diskstats returns {} when all lines are too short to parse.
        """
        short_line = "   8       0 sda 100\n"
        collector = DiskCollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data=short_line)):
            result = collector._parse_diskstats()  # pylint: disable=protected-access
        self.assertEqual(result, {})
    def test_parse_valid_diskstats(self):
        """
        Tests DiskCollector._parse_diskstats returns parsed device entries when the file is valid.
        """
        collector = DiskCollector(node_name="test-node")
        with patch("builtins.open", mock_open(read_data=SAMPLE_DISKSTATS)):
            result = collector._parse_diskstats()  # pylint: disable=protected-access

        self.assertIn("sda", result)
        self.assertIn("sda1", result)
        self.assertIn("loop0", result)

        sda = result["sda"]
        self.assertEqual(sda["reads_completed"], 100)
        self.assertEqual(sda["read_sectors"], 2000)
        self.assertEqual(sda["read_time_ms"], 500)
        self.assertEqual(sda["writes_completed"], 200)
        self.assertEqual(sda["write_sectors"], 4000)
        self.assertEqual(sda["write_time_ms"], 1000)


class TestCalculateIopsAndThroughput(unittest.TestCase):
    """Tests for DiskCollector._calculate_iops_and_throughput."""

    def test_no_history_returns_none(self):
        """
        Tests DiskCollector._calculate_iops_and_throughput returns (None, None, None, None) when no prior history exists.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        result = collector._calculate_iops_and_throughput(  # pylint: disable=protected-access
            "sda", 100, 200, 2000, 4000
        )
        self.assertEqual(result, (None, None, None, None))
        self.assertIn("sda", collector._io_history)  # pylint: disable=protected-access

    def test_with_history_returns_rates(self):
        """
        Tests DiskCollector._calculate_iops_and_throughput returns computed rates when history is available.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        collector._calculate_iops_and_throughput(
            "sda", 100, 200, 2000, 4000
        )  # pylint: disable=protected-access
        read_iops, write_iops, read_mb, write_mb = collector._calculate_iops_and_throughput(
            # pylint: disable=protected-access
            "sda", 130, 260, 2600, 5200
        )
        self.assertAlmostEqual(read_iops, 3.0)
        self.assertAlmostEqual(write_iops, 6.0)
        expected_read_mb = (600 * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * 10)
        expected_write_mb = (1200 * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * 10)
        self.assertAlmostEqual(read_mb, expected_read_mb)
        self.assertAlmostEqual(write_mb, expected_write_mb)

    def test_history_updated(self):
        """
        Tests DiskCollector._calculate_iops_and_throughput behavior when history is updated after successive calls.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        collector._calculate_iops_and_throughput(
            "sda", 100, 200, 2000, 4000
        )  # pylint: disable=protected-access
        collector._calculate_iops_and_throughput(
            "sda", 150, 250, 2500, 4500
        )  # pylint: disable=protected-access
        history = collector._io_history["sda"]  # pylint: disable=protected-access
        self.assertEqual(history["reads_completed"], 150)
        self.assertEqual(history["writes_completed"], 250)
        self.assertEqual(history["read_sectors"], 2500)
        self.assertEqual(history["write_sectors"], 4500)

    def test_counter_reset_returns_none(self):
        """
        Tests DiskCollector._calculate_iops_and_throughput returns (None, None, None, None) when
        counters decrease (counter reset or overflow), and updates history with the new values.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        collector._calculate_iops_and_throughput(  # pylint: disable=protected-access
            "sda", 1000, 2000, 10000, 20000
        )
        result = collector._calculate_iops_and_throughput(  # pylint: disable=protected-access
            "sda", 100, 200, 1000, 2000
        )
        self.assertEqual(result, (None, None, None, None))
        history = collector._io_history["sda"]  # pylint: disable=protected-access
        self.assertEqual(history["reads_completed"], 100)
        self.assertEqual(history["writes_completed"], 200)
        self.assertEqual(history["read_sectors"], 1000)
        self.assertEqual(history["write_sectors"], 2000)


class TestCalculateUtilization(unittest.TestCase):
    """Tests for DiskCollector._calculate_utilization."""

    def test_no_history_returns_none(self):
        """
        Tests DiskCollector._calculate_utilization returns (None, None) when no prior time history exists.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        collector._io_history["sda"] = {  # pylint: disable=protected-access
            "reads_completed": 0, "writes_completed": 0,
            "read_sectors": 0, "write_sectors": 0,
        }
        result = collector._calculate_utilization("sda", 500, 1000)  # pylint: disable=protected-access
        self.assertEqual(result, (None, None))

    def test_utilization_calculated_correctly(self):
        """
        Tests DiskCollector._calculate_utilization returns correct percentages when history contains prior time values.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        collector._io_history["sda"] = {"read_time_ms": 0, "write_time_ms": 0}  # pylint: disable=protected-access
        read_util, write_util = collector._calculate_utilization(
            "sda", 2000, 5000
        )  # pylint: disable=protected-access
        self.assertAlmostEqual(read_util, 20.0)
        self.assertAlmostEqual(write_util, 50.0)

    def test_counter_reset_returns_none(self):
        """
        Tests DiskCollector._calculate_utilization returns (None, None) when time counters
        decrease (counter reset), and updates history with the new values.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        collector._io_history["sda"] = {"read_time_ms": 5000, "write_time_ms": 8000}  # pylint: disable=protected-access
        read_util, write_util = collector._calculate_utilization(  # pylint: disable=protected-access
            "sda", 100, 200
        )
        self.assertIsNone(read_util)
        self.assertIsNone(write_util)
        history = collector._io_history["sda"]  # pylint: disable=protected-access
        self.assertEqual(history["read_time_ms"], 100)
        self.assertEqual(history["write_time_ms"], 200)

    def test_utilization_available_with_history(self):
        """
        Tests DiskCollector._calculate_utilization returns utilization metrics when two consecutive _collect_impl calls are made.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        first_data = "   8       0 sda 100 0 2000 500 200 0 4000 1000 0 800 1500 0 0 0 0\n"
        second_data = "   8       0 sda 130 0 2600 700 260 0 5200 1300 0 1000 1800 0 0 0 0\n"

        with patch("builtins.open", mock_open(read_data=first_data)):
            collector._collect_impl()  # pylint: disable=protected-access

        with patch("builtins.open", mock_open(read_data=second_data)):
            result = collector._collect_impl()  # pylint: disable=protected-access

        sda = result["sda"]
        self.assertIn("disk_read_utilization", sda)
        self.assertIn("disk_write_utilization", sda)
        # read_time delta=200ms in 10s window → 2%
        self.assertAlmostEqual(sda["disk_read_utilization"], 2.0)
        # write_time delta=300ms in 10s window → 3%
        self.assertAlmostEqual(sda["disk_write_utilization"], 3.0)


class TestCollectImpl(unittest.TestCase):
    """Tests for DiskCollector._collect_impl."""

    def test_collect_returns_empty_on_parse_error(self):
        """
        Tests DiskCollector._collect_impl returns {} when _parse_diskstats raises an unexpected error.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        with patch(
                "network_performance_exporter.collectors.collector_for_disk.DiskCollector._parse_diskstats",
                side_effect=RuntimeError("unexpected error"),
        ):
            result = collector._collect_impl()  # pylint: disable=protected-access
        self.assertEqual(result, {})

    def test_collect_removes_offline_devices_from_history(self):
        """
        Tests DiskCollector._collect_impl behavior when a device previously in history is no longer present in diskstats.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        collector._io_history["ghost"] = {  # pylint: disable=protected-access
            "reads_completed": 0, "writes_completed": 0,
            "read_sectors": 0, "write_sectors": 0,
        }
        with patch("builtins.open", mock_open(read_data=SAMPLE_DISKSTATS)):
            collector._collect_impl()  # pylint: disable=protected-access

        self.assertNotIn("ghost", collector._io_history)  # pylint: disable=protected-access

    def test_collect_returns_metrics_for_all_devices(self):
        """
        Tests DiskCollector._collect_impl returns metrics for all parsed devices on the first collection call.
        """
        collector = DiskCollector(node_name="test-node", update_interval=15)
        with patch("builtins.open", mock_open(read_data=SAMPLE_DISKSTATS)):
            result = collector._collect_impl()  # pylint: disable=protected-access

        self.assertIn("sda", result)
        self.assertIn("sda1", result)

        sda = result["sda"]
        self.assertEqual(sda["node"], "test-node")
        self.assertEqual(sda["device"], "sda")
        self.assertEqual(sda["disk_reads_completed_total"], 100)
        self.assertEqual(sda["disk_read_bytes_total"], 2000 * DiskStatsConstants.SECTOR_SIZE)
        self.assertEqual(sda["disk_read_time_ms_total"], 500)
        self.assertEqual(sda["disk_writes_completed_total"], 200)
        self.assertEqual(sda["disk_write_bytes_total"], 4000 * DiskStatsConstants.SECTOR_SIZE)
        self.assertEqual(sda["disk_write_time_ms_total"], 1000)
        # First call: no history yet, rate metrics should not be present
        self.assertNotIn("disk_read_iops", sda)
        self.assertNotIn("disk_write_iops", sda)
        self.assertNotIn("disk_read_throughput_mb", sda)
        self.assertNotIn("disk_write_throughput_mb", sda)
        self.assertNotIn("disk_read_utilization", sda)
        self.assertNotIn("disk_write_utilization", sda)

        # history initialized with current snapshot values
        history = collector._io_history["sda"]  # pylint: disable=protected-access
        self.assertEqual(history["reads_completed"], 100)
        self.assertEqual(history["writes_completed"], 200)
        self.assertEqual(history["read_sectors"], 2000)
        self.assertEqual(history["write_sectors"], 4000)
        self.assertEqual(history["read_time_ms"], 500)
        self.assertEqual(history["write_time_ms"], 1000)

    def test_collect_rates_with_history(self):
        """
        Tests DiskCollector._collect_impl returns computed rate and utilization metrics when history is available from a prior call.
        """
        collector = DiskCollector(node_name="test-node", update_interval=10)
        first_data = "   8       0 sda 100 0 2000 500 200 0 4000 1000 0 800 1500 0 0 0 0\n"
        second_data = "   8       0 sda 130 0 2600 700 260 0 5200 1300 0 1000 1800 0 0 0 0\n"

        with patch("builtins.open", mock_open(read_data=first_data)):
            collector._collect_impl()  # pylint: disable=protected-access

        with patch("builtins.open", mock_open(read_data=second_data)):
            result = collector._collect_impl()  # pylint: disable=protected-access

        sda = result["sda"]
        # totals reflect current snapshot
        self.assertEqual(sda["disk_reads_completed_total"], 130)
        self.assertEqual(sda["disk_read_bytes_total"], 2600 * DiskStatsConstants.SECTOR_SIZE)
        self.assertEqual(sda["disk_read_time_ms_total"], 700)
        self.assertEqual(sda["disk_writes_completed_total"], 260)
        self.assertEqual(sda["disk_write_bytes_total"], 5200 * DiskStatsConstants.SECTOR_SIZE)
        self.assertEqual(sda["disk_write_time_ms_total"], 1300)
        # iops: delta 30 reads / 10s, 60 writes / 10s
        self.assertAlmostEqual(sda["disk_read_iops"], 3.0)
        self.assertAlmostEqual(sda["disk_write_iops"], 6.0)
        # throughput: delta 600 / 1200 sectors in 10s
        expected_read_mb = (600 * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * 10)
        expected_write_mb = (1200 * DiskStatsConstants.SECTOR_SIZE) / (1024 * 1024 * 10)
        self.assertAlmostEqual(sda["disk_read_throughput_mb"], expected_read_mb)
        self.assertAlmostEqual(sda["disk_write_throughput_mb"], expected_write_mb)
        # utilization: read_time delta=200ms, write_time delta=300ms in 10s window
        self.assertAlmostEqual(sda["disk_read_utilization"], 2.0)
        self.assertAlmostEqual(sda["disk_write_utilization"], 3.0)

        # history updated with second round values
        history = collector._io_history["sda"]  # pylint: disable=protected-access
        self.assertEqual(history["reads_completed"], 130)
        self.assertEqual(history["writes_completed"], 260)
        self.assertEqual(history["read_sectors"], 2600)
        self.assertEqual(history["write_sectors"], 5200)
        self.assertEqual(history["read_time_ms"], 700)
        self.assertEqual(history["write_time_ms"], 1300)


if __name__ == "__main__":
    unittest.main()
