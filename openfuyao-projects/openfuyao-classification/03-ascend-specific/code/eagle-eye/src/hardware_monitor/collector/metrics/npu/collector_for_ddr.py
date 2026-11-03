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
This module defines DDRCollector class.
"""

import logging
import os

from ctypes import byref, c_int
from typing import Any

from common.base.base_metrics_collector import BaseCollector
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.dcmi.dcmi_wrapper import DcmiMemoryInfoStruct
from common.npu_metrics.constants import (RET_INVALID_VALUE, RET_NO_FUNCTION, ASCEND_910B,
                                          ASCEND_910A3)
from common.npu_metrics.utils import fetch_device_info_and_type
from hardware_monitor.base.registry import registry
from hardware_monitor.base.utils import add_error, enrich_result_with_container_info
from hardware_monitor.collector.metrics.npu.npu_utils import init_device_info

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class DDRCollector(BaseCollector):
    """
    This class collects DDR information.
    """
    @property
    def name(self):
        return "ddr"

    def _collect_impl(self):
        results = {}
        node_name = os.getenv("NODE_NAME")
        logic_ids = self.cache.get("logic_id") or []
        phy_ids = self.cache.get("phy_id") or []
        card_ids = self.cache.get("card_id") or []
        device_ids = self.cache.get("device_id") or []
        for i, logic_id in enumerate(logic_ids):
            phy_id = phy_ids[i]
            card_id = card_ids[i]
            device_id = device_ids[i]
            info = self._get_device_all_ddr_info(node_name, logic_id, phy_id, card_id, device_id)
            results[str(logic_id)] = enrich_result_with_container_info(info, phy_id, self.cache)
        return results

    @staticmethod
    def _get_device_all_ddr_info(node_name: str, logic_id: int, phy_id: int,
                                 card_id: int, device_id: int) -> dict[str, Any]:
        fields = [
            "node_name", "phy_id", "card_id", "device_id",
            "namespace", "pod_name", "container_name",
            "npu_chip_info_total_memory", "npu_chip_info_used_memory",
            "npu_chip_info_memory_frequency", "npu_chip_info_memory_utilization"
        ]

        result = init_device_info(node_name, phy_id, card_id, device_id, fields)

        basic_info = DDRCollector._get_device_basic_info(logic_id, card_id, device_id)
        if "error_code" in basic_info:
            add_error(result, "mem_info", basic_info["error_code"], basic_info["error_message"])
            return result

        mem_info = DDRCollector._get_device_memory_info(logic_id, card_id, device_id)
        if "error_code" not in mem_info:
            result.update({
                "npu_chip_info_total_memory": mem_info["memory_size"],
                "npu_chip_info_used_memory": mem_info["memory_size"] - mem_info["memory_available"],
                "npu_chip_info_memory_frequency": mem_info["freq"],
                "npu_chip_info_memory_utilization": mem_info["utiliza"]
            })
        else:
            add_error(result, "mem_info", mem_info["error_code"], mem_info["error_message"])

        return result

    @staticmethod
    def _get_device_basic_info(logic_id: int, card_id: int, device_id: int):
        _, _, device_type, code, err = fetch_device_info_and_type(card_id, device_id)
        if err is not None:
            msg = f"Failed to resolve device type from logicID({logic_id}): {err}"
            logger.error("[DDRCollector] %s",msg)
            return {"error_code": code, "error_message": msg}

        if device_type in (ASCEND_910B, ASCEND_910A3):
            msg = f"LogicID({logic_id}): {device_type} does not have ddr module"
            logger.error("[DDRCollector] %s",msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {}

    @staticmethod
    def _get_device_memory_info(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_memory_info_v3"):
            msg = "dcmi_get_device_memory_info_v3 not available in shared object."
            logger.error("[DDRCollector] %s",msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        mem_info = DcmiMemoryInfoStruct()
        ret = dcmi_instance.dcmi.dcmi_get_device_memory_info_v3(
            c_int(card_id), c_int(device_id), byref(mem_info)
        )
        if ret != 0:
            msg = f"Failed to get memory info for logicID({logic_id}), ret: {ret}."
            logger.error("[DDRCollector] %s",msg)
            return {"error_code": ret, "error_message": msg}

        total = int(mem_info.memory_size)
        available = int(mem_info.memory_available)
        if total == 0 or available > total:
            msg = (f"Invalid memory info for logicID({logic_id}), "
                   f"total({total}) < available({available}) or total is 0")
            logger.error("[DDRCollector] %s",msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {
            "memory_size": int(mem_info.memory_size),
            "memory_available": int(mem_info.memory_available),
            "freq": int(mem_info.freq),
            "hugepagesize": int(mem_info.hugepagesize),
            "hugepages_total": int(mem_info.hugepages_total),
            "hugepages_free": int(mem_info.hugepages_free),
            "utiliza": int(mem_info.utiliza),
        }


collector = DDRCollector()
registry.register(collector.name, collector, group="collector.npu")
