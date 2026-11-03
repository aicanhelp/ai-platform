# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for hccn_tools."""

import unittest

from unittest.mock import patch

from common.npu_metrics.constants import RET_INVALID_VALUE
from common.npu_tools.hccn_tools import (
    get_npu_link_status,
    get_npu_interface_traffic,
    get_npu_link_up_count,
    get_npu_link_speed,
    get_npu_packet_statistics,
)


def _mock_hccn(output: str):
    """Patch get_info_from_hccn_tool to return the given output."""
    return patch(
        "common.npu_tools.hccn_tools.get_info_from_hccn_tool",
        return_value=(output, None, None),
    )


def _mock_hccn_error(ret_code: int, err: Exception):
    """Patch get_info_from_hccn_tool to return an error."""
    return patch(
        "common.npu_tools.hccn_tools.get_info_from_hccn_tool",
        return_value=("", ret_code, err),
    )


class TestGetNpuPacketStatistics(unittest.TestCase):
    """Tests for get_npu_packet_statistics."""

    def test_hccn_error_returns_error(self):
        """
        Tests get_npu_packet_statistics returns error_code when hccn_tool fails.
        """
        with _mock_hccn_error(-1, RuntimeError("hccn_tool failed")):
            result = get_npu_packet_statistics(0, 0)
        self.assertIn("error_code", result)

    def test_no_relevant_fields_returns_error(self):
        """
        Tests get_npu_packet_statistics returns error_code when no relevant fields are found.
        """
        with _mock_hccn("unrelated_field:123\n"):
            result = get_npu_packet_statistics(0, 0)
        self.assertIn("error_code", result)
        self.assertEqual(result["error_code"], RET_INVALID_VALUE)

    def test_returns_all_fields(self):
        """
        Tests get_npu_packet_statistics returns all relevant fields when output is valid.
        """
        output = (
            "packet statistics:\n"
            "roce_rx_err_pkt_num:10\n"
            "roce_rx_all_pkt_num:1000\n"
            "roce_new_pkt_rty_num:5\n"
            "roce_tx_all_pkt_num:500\n"
        )
        with _mock_hccn(output):
            result = get_npu_packet_statistics(0, 0)
        self.assertEqual(result["roce_rx_err_pkt_num"], 10)
        self.assertEqual(result["roce_rx_all_pkt_num"], 1000)
        self.assertEqual(result["roce_new_pkt_rty_num"], 5)
        self.assertEqual(result["roce_tx_all_pkt_num"], 500)

    def test_partial_fields_returned(self):
        """
        Tests get_npu_packet_statistics returns only available fields when output is partial.
        """
        output = "roce_rx_err_pkt_num:10\nroce_rx_all_pkt_num:1000\n"
        with _mock_hccn(output):
            result = get_npu_packet_statistics(0, 0)
        self.assertIn("roce_rx_err_pkt_num", result)
        self.assertIn("roce_rx_all_pkt_num", result)
        self.assertNotIn("roce_new_pkt_rty_num", result)


class TestGetNpuLinkSpeed(unittest.TestCase):
    """Tests for get_npu_link_speed."""

    def test_hccn_error_returns_error(self):
        """
        Tests get_npu_link_speed returns error_code when hccn_tool fails.
        """
        with _mock_hccn_error(-1, RuntimeError("hccn_tool failed")):
            result = get_npu_link_speed(0, 0)
        self.assertIn("error_code", result)

    def test_unrecognized_output_returns_error(self):
        """
        Tests get_npu_link_speed returns error_code when output format is unrecognized.
        """
        with _mock_hccn("no speed info\n"):
            result = get_npu_link_speed(0, 0)
        self.assertIn("error_code", result)
        self.assertEqual(result["error_code"], RET_INVALID_VALUE)

    def test_returns_speed_in_mb(self):
        """
        Tests get_npu_link_speed returns speed converted to MB/s when output is valid.
        """
        with _mock_hccn("Speed: 200000 Mb/s\n"):
            result = get_npu_link_speed(0, 0)
        # 200000 Mb/s * 0.125 = 25000 MB/s
        self.assertAlmostEqual(result["speed"], 25000.0)


class TestGetNpuInterfaceTraffic(unittest.TestCase):
    """Tests for get_npu_interface_traffic."""

    def test_hccn_error_returns_error(self):
        """
        Tests get_npu_interface_traffic returns error_code when hccn_tool fails.
        """
        with _mock_hccn_error(-1, RuntimeError("hccn_tool failed")):
            result = get_npu_interface_traffic(0, 0)
        self.assertIn("error_code", result)

    def test_missing_tx_returns_error(self):
        """
        Tests get_npu_interface_traffic returns error_code when TX is missing from output.
        """
        with _mock_hccn("Bandwidth RX: 200.0 MB/sec\n"):
            result = get_npu_interface_traffic(0, 0)
        self.assertIn("error_code", result)

    def test_missing_rx_returns_error(self):
        """
        Tests get_npu_interface_traffic returns error_code when RX is missing from output.
        """
        with _mock_hccn("Bandwidth TX: 100.0 MB/sec\n"):
            result = get_npu_interface_traffic(0, 0)
        self.assertIn("error_code", result)

    def test_returns_tx_rx(self):
        """
        Tests get_npu_interface_traffic returns tx and rx values when output is valid.
        """
        output = "Bandwidth TX: 100.5 MB/sec\nBandwidth RX: 200.0 MB/sec\n"
        with _mock_hccn(output):
            result = get_npu_interface_traffic(0, 0)
        self.assertAlmostEqual(result["tx"], 100.5)
        self.assertAlmostEqual(result["rx"], 200.0)


class TestGetNpuLinkUpCount(unittest.TestCase):
    """Tests for get_npu_link_up_count."""

    def test_hccn_error_returns_error(self):
        """
        Tests get_npu_link_up_count returns error_code when hccn_tool fails.
        """
        with _mock_hccn_error(-1, RuntimeError("hccn_tool failed")):
            result = get_npu_link_up_count(0, 0)
        self.assertIn("error_code", result)

    def test_returns_count(self):
        """
        Tests get_npu_link_up_count returns link_up_count when output is valid.
        """
        with _mock_hccn("[devid 0]link up count : 5\n"):
            result = get_npu_link_up_count(0, 0)
        self.assertEqual(result["link_up_count"], 5)


class TestGetNpuLinkStatus(unittest.TestCase):
    """Tests for get_npu_link_status."""

    def test_hccn_error_returns_error(self):
        """
        Tests get_npu_link_status returns error_code when hccn_tool fails.
        """
        with _mock_hccn_error(-1, RuntimeError("hccn_tool failed")):
            result = get_npu_link_status(0, 0)
        self.assertIn("error_code", result)
        self.assertIn("error_message", result)

    def test_link_up(self):
        """
        Tests get_npu_link_status returns status UP when hccn_tool reports link UP.
        """
        with _mock_hccn("link status: UP\n"):
            result = get_npu_link_status(0, 0)
        self.assertEqual(result["status"], "UP")

    def test_link_down(self):
        """
        Tests get_npu_link_status returns status DOWN when hccn_tool reports link DOWN.
        """
        with _mock_hccn("link status: DOWN\n"):
            result = get_npu_link_status(0, 0)
        self.assertEqual(result["status"], "DOWN")


if __name__ == "__main__":
    unittest.main()
