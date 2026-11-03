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
This module defines NetworkCollector class with parallel execution optimization.

Key improvement: get_npu_interface_traffic() and get_npu_link_status() are executed
in parallel using ThreadPoolExecutor, reducing collection time by ~50%.
"""

import logging
import os
import time

from typing import Any
from concurrent.futures import as_completed

from common.base.base_metrics_collector import BaseCollector
from common.npu_metrics.constants import (RET_INVALID_VALUE, ASCEND_310, ASCEND_910B,
                                          A300I_A2_Board_Id, A300I_A2_GB64_Board_Id)
from common.npu_tools.hccn_tools import get_npu_interface_traffic, get_npu_link_status
from common.npu_metrics.utils import fetch_device_info_and_type
from hardware_monitor.base.registry import registry
from hardware_monitor.base.utils import add_error, enrich_result_with_container_info
from hardware_monitor.collector.metrics.npu.npu_utils import init_device_info

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)

# Timeout for individual operations (in seconds)
OPERATION_TIMEOUT = 6


class NetworkCollector(BaseCollector):
    """
    This collector collects information about the network.

    Key optimization: Parallel execution of get_npu_interface_traffic() and
    get_npu_link_status() using ThreadPoolExecutor with as_completed().
    """
    @property
    def name(self):
        return "network"

    def _collect_impl(self):
        results = {}
        node_name = os.getenv("NODE_NAME", "")
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
            info = self._get_device_all_network_info(node_name, logic_id, phy_id, card_id, device_id, timeout=remaining)
            results[str(logic_id)] = enrich_result_with_container_info(info, phy_id, self.cache)
        return results

    def _get_device_all_network_info(self, node_name: str, logic_id: int, phy_id: int,
                                     card_id: int, device_id: int, timeout: float) -> dict[str, Any]:
        fields = [
            "node_name", "phy_id", "card_id", "device_id",
            "namespace", "pod_name", "container_name",
            "npu_chip_info_bandwidth_tx", "npu_chip_info_bandwidth_rx", "npu_chip_info_link_status"
        ]

        result = init_device_info(node_name, phy_id, card_id, device_id, fields)

        basic_info = self._get_device_basic_info(logic_id, card_id, device_id)
        if "error_code" in basic_info:
            add_error(result, "mem_info", basic_info["error_code"], basic_info["error_message"])
            return result

        # Parallel execution of network metrics collection using as_completed()
        futures = {
            self.executor.submit(get_npu_interface_traffic, logic_id, phy_id): "traffic",
            self.executor.submit(get_npu_link_status, logic_id, phy_id): "link_status",
        }

        metrics = {}
        try:
            for future in as_completed(futures, timeout=timeout):
                key = futures[future]
                try:
                    metrics[key] = future.result()
                except Exception as e:
                    logger.error(
                        "Failed to collect %s for logic_id=%d, phy_id=%d: %s",
                        key, logic_id, phy_id, str(e)
                    )
                    metrics[key] = {"error_code": RET_INVALID_VALUE, "error_message": str(e)}
        except Exception as e:
            logger.error(
                "Parallel collection timeout for logic_id=%d, phy_id=%d",
                logic_id, phy_id
            )
            for key in futures.values():
                if key not in metrics:
                    metrics[key] = {"error_code": RET_INVALID_VALUE, "error_message": "Timeout"}

        # Process traffic data
        npu_traffic = metrics.get("traffic", {})
        if "error_code" not in npu_traffic:
            result.update({
                "npu_chip_info_bandwidth_tx": npu_traffic["tx"],
                "npu_chip_info_bandwidth_rx": npu_traffic["rx"]})
        else:
            add_error(result, "network_info", npu_traffic["error_code"], npu_traffic["error_message"])

        # Process link status data
        npu_link_status = metrics.get("link_status", {})
        if "error_code" not in npu_link_status:
            result.update({
                "npu_chip_info_link_status": npu_link_status["status"]})
        else:
            add_error(result, "network_info", npu_link_status["error_code"], npu_link_status["error_message"])

        return result

    @staticmethod
    def _get_device_basic_info(logic_id: int, card_id: int, device_id: int):
        _, board_info, device_type, code, err = fetch_device_info_and_type(card_id, device_id)
        if err is not None:
            msg = f"Failed to resolve device type from logicID({logic_id}): {err}"
            logger.error("[NetworkCollector] %s", msg)
            return {"error_code": code, "error_message": msg}

        if device_type.startswith(ASCEND_310) or (device_type == ASCEND_910B and
                                                  board_info["board_id"] in (A300I_A2_Board_Id, A300I_A2_GB64_Board_Id)):
            msg = f"LogicID({logic_id}): {device_type} is not a training card"
            logger.error("[NetworkCollector] %s", msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {}


collector = NetworkCollector()
registry.register(collector.name, collector, group="collector.npu")
