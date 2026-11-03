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
NPU RoCE network metrics collector.
"""

import logging
import os
import time

from concurrent.futures import as_completed
from typing import Any, Dict

from common.base.base_metrics_collector import BaseCollector
from common.npu_metrics.constants import (ASCEND_310, ASCEND_910B,
                                          A300I_A2_Board_Id, A300I_A2_GB64_Board_Id)
from common.npu_metrics.utils import fetch_device_info_and_type
from common.npu_tools.hccn_tools import (
    get_npu_link_status,
    get_npu_interface_traffic,
    get_npu_link_up_count,
    get_npu_link_speed,
    get_npu_packet_statistics,
)

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class NPURoCECollector(BaseCollector):
    """
    Collector for NPU RoCE network metrics using shared thread pool.
    """

    def __init__(self, node_name: str = "", update_interval: int = 15):
        super().__init__(update_interval=update_interval)
        self.node_name = node_name or os.getenv("NODE_NAME", "")
        logger.info(
            "NPURoCECollector initialized with update_interval=%d seconds (shared thread pool)",
            update_interval
        )

    @property
    def name(self) -> str:
        """Get collector name."""
        return "npu_roce"

    def _collect_impl(self) -> Dict[str, Any]:
        """
        Collect NPU RoCE network metrics from cache device info.

        Returns:
            Dictionary with NPU RoCE metrics keyed by logic_id
        """
        result = {}
        logic_ids = self.cache.get("logic_id") or []
        phy_ids = self.cache.get("phy_id") or []
        card_ids = self.cache.get("card_id") or []
        device_ids = self.cache.get("device_id") or []

        start_time = time.time()

        for i, logic_id in enumerate(logic_ids):
            elapsed = time.time() - start_time
            remaining = self.update_interval - elapsed

            if remaining <= 0:
                logger.warning("Timeout: no time left for remaining devices")
                break

            phy_id = phy_ids[i]
            card_id = card_ids[i]
            device_id = device_ids[i]

            metrics = self._collect_npu_metrics(logic_id, phy_id, card_id, device_id, timeout=remaining)
            if any(k not in ("node", "logic_id", "phy_id", "card_id", "device_id") for k in metrics):
                result[logic_id] = metrics

        return result

    def _collect_npu_metrics(self, logic_id: int, phy_id: int, card_id: int, device_id: int, timeout: float) -> Dict[str, Any]:
        """
        Collect metrics for a specific NPU using shared thread pool with parallel execution.

        Args:
            logic_id: Logical NPU ID
            phy_id: Physical NPU ID
            card_id: Card ID
            device_id: Device ID
            timeout: Timeout in seconds for this device's collection

        Returns:
            Dictionary with NPU RoCE metrics
        """
        metrics = {
            "node": self.node_name,
            "logic_id": logic_id,
            "phy_id": phy_id,
            "card_id": card_id,
            "device_id": device_id,
        }

        _, board_info, device_type, code, err = fetch_device_info_and_type(card_id, device_id)

        if err is not None:
            logger.warning("Failed to get device info for logic_id=%d, card_id=%d, device_id=%d: %s",
                           logic_id, card_id, device_id, err)
            return metrics

        if device_type.startswith(ASCEND_310) or (device_type == ASCEND_910B and
                                                  board_info["board_id"] in (A300I_A2_Board_Id, A300I_A2_GB64_Board_Id)):
            logger.debug("logic_id=%d device_type=%s is not a training card, skipping hccn_tool collection",
                         logic_id, device_type)
            return metrics

        futures = {
            self.executor.submit(get_npu_link_status, logic_id, phy_id): "link_status",
            self.executor.submit(get_npu_link_up_count, logic_id, phy_id): "link_up_count",
            self.executor.submit(get_npu_interface_traffic, logic_id, phy_id): "traffic",
            self.executor.submit(get_npu_link_speed, logic_id, phy_id): "speed",
            self.executor.submit(get_npu_packet_statistics, logic_id, phy_id): "stats",
        }

        results = {}
        try:
            for future in as_completed(futures, timeout=timeout):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(
                        "Failed to collect %s for logic_id=%d, phy_id=%d: %s",
                        key, logic_id, phy_id, str(e)
                    )
        except Exception as e:
            logger.error(
                "Parallel collection timeout or error for logic_id=%d: %s",
                logic_id, str(e)
            )

        if "link_status" in results and "error_code" not in results["link_status"]:
            status = results["link_status"].get("status", "").upper()
            metrics["npu_roce_link_state"] = 1 if status == "UP" else 0

        if "link_up_count" in results and "error_code" not in results["link_up_count"]:
            metrics["npu_roce_link_up_count"] = results["link_up_count"].get("link_up_count")

        if "traffic" in results and "error_code" not in results["traffic"]:
            metrics["npu_roce_tx_rate_mbps"] = results["traffic"].get("tx")
            metrics["npu_roce_rx_rate_mbps"] = results["traffic"].get("rx")

        if "speed" in results and "error_code" not in results["speed"]:
            speed = results["speed"].get("speed")
            if speed is not None and "npu_roce_tx_rate_mbps" in metrics:
                tx_rate = metrics["npu_roce_tx_rate_mbps"]
                metrics["npu_roce_tx_available_bandwidth_mbps"] = max(0.0, speed - tx_rate)
            if speed is not None and "npu_roce_rx_rate_mbps" in metrics:
                rx_rate = metrics["npu_roce_rx_rate_mbps"]
                metrics["npu_roce_rx_available_bandwidth_mbps"] = max(0.0, speed - rx_rate)

        if "stats" in results and "error_code" not in results["stats"]:
            stats_response = results["stats"]

            if "roce_rx_err_pkt_num" in stats_response and "roce_rx_all_pkt_num" in stats_response:
                rx_err = stats_response["roce_rx_err_pkt_num"]
                rx_all = stats_response["roce_rx_all_pkt_num"]
                if rx_all > 0:
                    packet_loss_rate = rx_err / rx_all
                    metrics["npu_roce_packet_loss_rate"] = packet_loss_rate

            if "roce_new_pkt_rty_num" in stats_response and "roce_tx_all_pkt_num" in stats_response:
                rty_num = stats_response["roce_new_pkt_rty_num"]
                tx_all = stats_response["roce_tx_all_pkt_num"]
                if tx_all > 0:
                    retransmit_rate = rty_num / tx_all
                    metrics["npu_roce_retransmit_rate"] = retransmit_rate

        return metrics
