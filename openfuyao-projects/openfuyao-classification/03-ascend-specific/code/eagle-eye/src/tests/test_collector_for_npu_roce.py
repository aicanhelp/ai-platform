# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for NPURoCECollector."""

import unittest

from unittest.mock import patch

from network_performance_exporter.collectors.collector_for_npu_roce import NPURoCECollector

_ERROR = {"error_code": -1, "error_message": "not provided"}


def _make_collector():
    return NPURoCECollector(node_name="test-node", update_interval=15)


class TestCollectNpuMetrics(unittest.TestCase):
    """Tests for NPURoCECollector._collect_npu_metrics."""

    def _run(self, results: dict) -> dict:
        collector = _make_collector()
        base = "network_performance_exporter.collectors.collector_for_npu_roce"
        with patch(f"{base}.fetch_device_info_and_type",
                   return_value=({}, {"board_id": 0xb0}, "ASCEND_910B", None, None)), \
                patch(f"{base}.get_npu_link_status",
                      return_value=results.get("link_status", _ERROR)), \
                patch(f"{base}.get_npu_link_up_count",
                      return_value=results.get("link_up_count", _ERROR)), \
                patch(f"{base}.get_npu_interface_traffic",
                      return_value=results.get("traffic", _ERROR)), \
                patch(f"{base}.get_npu_link_speed",
                      return_value=results.get("speed", _ERROR)), \
                patch(f"{base}.get_npu_packet_statistics",
                      return_value=results.get("stats", _ERROR)):
            return collector._collect_npu_metrics(0, 0, 0, 0, timeout=15)  # pylint: disable=protected-access

    def test_link_status_error_omits_field(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when the link status future returns an error and the field is omitted.
        """
        metrics = self._run({"link_status": {"error_code": -1, "error_message": "failed"}})
        self.assertNotIn("npu_roce_link_state", metrics)

    def test_link_status_up(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns link state 1 when the link status is UP.
        """
        metrics = self._run({"link_status": {"status": "UP"}})
        self.assertEqual(metrics["npu_roce_link_state"], 1)

    def test_link_status_down(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns link state 0 when the link status is DOWN.
        """
        metrics = self._run({"link_status": {"status": "DOWN"}})
        self.assertEqual(metrics["npu_roce_link_state"], 0)

    def test_link_up_count_error_omits_field(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when the link up count future returns an error and the field is omitted.
        """
        metrics = self._run({"link_up_count": {"error_code": -1, "error_message": "failed"}})
        self.assertNotIn("npu_roce_link_up_count", metrics)

    def test_link_up_count(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns the link up count when the future succeeds.
        """
        metrics = self._run({"link_up_count": {"link_up_count": 5}})
        self.assertEqual(metrics["npu_roce_link_up_count"], 5)

    def test_traffic_error_omits_fields(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when the traffic future returns an error and rate fields are omitted.
        """
        metrics = self._run({"traffic": {"error_code": -1, "error_message": "failed"}})
        self.assertNotIn("npu_roce_tx_rate_mbps", metrics)
        self.assertNotIn("npu_roce_rx_rate_mbps", metrics)

    def test_traffic(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns tx and rx rate metrics when the traffic future succeeds.
        """
        metrics = self._run({"traffic": {"tx": 100.0, "rx": 200.0}})
        self.assertEqual(metrics["npu_roce_tx_rate_mbps"], 100.0)
        self.assertEqual(metrics["npu_roce_rx_rate_mbps"], 200.0)

    def test_available_bandwidth(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns available bandwidth metrics when both traffic and speed futures succeed.
        """
        metrics = self._run({
            "traffic": {"tx": 100.0, "rx": 200.0},
            "speed": {"speed": 1000.0},
        })
        self.assertAlmostEqual(metrics["npu_roce_tx_available_bandwidth_mbps"], 900.0)
        self.assertAlmostEqual(metrics["npu_roce_rx_available_bandwidth_mbps"], 800.0)

    def test_stats_error_omits_fields(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when the stats future returns an error and packet rate fields are omitted.
        """
        metrics = self._run({"stats": {"error_code": -1, "error_message": "failed"}})
        self.assertNotIn("npu_roce_packet_loss_rate", metrics)
        self.assertNotIn("npu_roce_retransmit_rate", metrics)

    def test_available_bandwidth_no_traffic_omits_field(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when traffic is missing and available bandwidth fields are omitted.
        """
        metrics = self._run({"speed": {"speed": 1000.0}})
        self.assertNotIn("npu_roce_tx_available_bandwidth_mbps", metrics)
        self.assertNotIn("npu_roce_rx_available_bandwidth_mbps", metrics)

    def test_packet_loss_rate_zero_rx_all_omits_field(self):
        """
        Tests NPURoCECollector._collect_npu_metrics behavior when total packet counts are zero and rate fields are omitted.
        """
        metrics = self._run({"stats": {
            "roce_rx_err_pkt_num": 0,
            "roce_rx_all_pkt_num": 0,
            "roce_new_pkt_rty_num": 0,
            "roce_tx_all_pkt_num": 0,
        }})
        self.assertNotIn("npu_roce_packet_loss_rate", metrics)
        self.assertNotIn("npu_roce_retransmit_rate", metrics)

    def test_packet_loss_rate(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns packet loss and retransmit rates when stats are valid.
        """
        metrics = self._run({"stats": {
            "roce_rx_err_pkt_num": 10,
            "roce_rx_all_pkt_num": 1000,
            "roce_new_pkt_rty_num": 5,
            "roce_tx_all_pkt_num": 500,
        }})
        self.assertAlmostEqual(metrics["npu_roce_packet_loss_rate"], 0.01)
        self.assertAlmostEqual(metrics["npu_roce_retransmit_rate"], 0.01)

    def test_base_fields_always_present(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns base identification fields regardless of future results.
        """
        metrics = self._run({})
        self.assertEqual(metrics["node"], "test-node")
        self.assertEqual(metrics["logic_id"], 0)
        self.assertEqual(metrics["phy_id"], 0)
        self.assertEqual(metrics["card_id"], 0)
        self.assertEqual(metrics["device_id"], 0)

    def test_device_info_error_returns_base_fields_only(self):
        """
        Tests NPURoCECollector._collect_npu_metrics returns only base fields when fetch_device_info_and_type fails.
        """
        collector = _make_collector()
        base = "network_performance_exporter.collectors.collector_for_npu_roce"
        with patch(f"{base}.fetch_device_info_and_type",
                   return_value=({}, {}, None, -1, RuntimeError("dcmi error"))):
            metrics = collector._collect_npu_metrics(0, 0, 0, 0, timeout=15)  # pylint: disable=protected-access
        self.assertEqual(metrics["node"], "test-node")
        self.assertNotIn("npu_roce_link_state", metrics)

    def test_non_training_card_skips_hccn(self):
        """
        Tests NPURoCECollector._collect_npu_metrics skips hccn_tool calls for non-training cards.
        """
        collector = _make_collector()
        base = "network_performance_exporter.collectors.collector_for_npu_roce"
        with patch(f"{base}.fetch_device_info_and_type",
                   return_value=({}, {"board_id": 0}, "Ascend310", None, None)), \
                patch(f"{base}.get_npu_link_status") as mock_link:
            metrics = collector._collect_npu_metrics(0, 0, 0, 0, timeout=15)  # pylint: disable=protected-access
        mock_link.assert_not_called()
        self.assertNotIn("npu_roce_link_state", metrics)


class TestCollectImpl(unittest.TestCase):
    """Tests for NPURoCECollector._collect_impl."""

    def test_collect_iterates_all_logic_ids(self):
        """
        Tests NPURoCECollector._collect_impl behavior when it iterates over all logic IDs in the cache.
        """
        collector = _make_collector()
        collector.cache = {
            "logic_id": [0, 1],
            "phy_id": [0, 1],
            "card_id": [0, 0],
            "device_id": [0, 1],
        }
        with patch.object(collector, "_collect_npu_metrics",
                          return_value={"node": "test-node", "npu_roce_link_state": 1}) as mock_collect:
            result = collector._collect_impl()  # pylint: disable=protected-access

        self.assertEqual(mock_collect.call_count, 2)
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_collect_empty_cache_returns_empty(self):
        """
        Tests NPURoCECollector._collect_impl returns {} when the cache contains no logic IDs.
        """
        collector = _make_collector()
        collector.cache = {}
        result = collector._collect_impl()  # pylint: disable=protected-access
        self.assertEqual(result, {})

    def test_collect_timeout_stops_early(self):
        """
        Tests NPURoCECollector._collect_impl behavior when the update interval elapses and collection stops before processing all logic IDs.
        """
        collector = NPURoCECollector(node_name="test-node", update_interval=1)
        collector.cache = {
            "logic_id": [0, 1, 2],
            "phy_id": [0, 1, 2],
            "card_id": [0, 0, 0],
            "device_id": [0, 1, 2],
        }

        call_count = 0

        def slow_collect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"node": "test-node", "npu_roce_link_state": 1}

        with patch("network_performance_exporter.collectors.collector_for_npu_roce.time") as mock_time:
            # First call: elapsed=0, remaining=1
            # Second call: elapsed=2 > update_interval=1, should stop
            mock_time.time.side_effect = [0.0, 0.0, 2.0]
            with patch.object(collector, "_collect_npu_metrics", side_effect=slow_collect):
                result = collector._collect_impl()  # pylint: disable=protected-access

        self.assertEqual(call_count, 1)
        self.assertIn(0, result)
        self.assertNotIn(1, result)


if __name__ == "__main__":
    unittest.main()
