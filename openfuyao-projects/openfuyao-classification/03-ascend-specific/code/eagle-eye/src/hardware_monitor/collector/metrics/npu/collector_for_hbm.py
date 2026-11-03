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
This module defines HBMCollector class.
"""

import logging
import os
from ctypes import byref, c_int
from typing import Any

from common.base.base_metrics_collector import BaseCollector
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.dcmi.dcmi_wrapper import DcmiHBMInfoStruct
from common.npu_metrics.constants import (RET_INVALID_VALUE, RET_NO_FUNCTION, HBM_DEVICE_TYPE_CODE,
                                          ASCEND_910, ASCEND_910B, ASCEND_910A3, ASCEND_910A5)
from common.npu_metrics.utils import fetch_device_info_and_type
from hardware_monitor.base.registry import registry
from hardware_monitor.base.utils import add_error, enrich_result_with_container_info
from hardware_monitor.collector.metrics.npu.npu_utils import init_device_info, get_utilization_rate

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class HBMCollector(BaseCollector):
    """
    This class collect HBM Data.
    """

    @property
    def name(self):
        return "hbm"

    def _collect_impl(self):
        results = {}
        node_name = os.getenv("NODE_NAME", "")
        logic_ids = self.cache.get("logic_id") or []
        phy_ids = self.cache.get("phy_id") or []
        card_ids = self.cache.get("card_id") or []
        device_ids = self.cache.get("device_id") or []
        for i, logic_id in enumerate(logic_ids):
            phy_id = phy_ids[i]
            card_id = card_ids[i]
            device_id = device_ids[i]
            info = self._get_device_all_hbm_info(node_name, logic_id, phy_id, card_id, device_id)
            results[str(logic_id)] = enrich_result_with_container_info(info, phy_id, self.cache)
        return results

    @staticmethod
    def _get_device_all_hbm_info(node_name: str, logic_id: int, phy_id: int,
                                 card_id: int, device_id: int) -> dict[str, Any]:
        fields = [
            "node_name", "phy_id", "card_id", "device_id",
            "namespace", "pod_name", "container_name",
            "npu_chip_info_hbm_total_memory", "npu_chip_info_hbm_frequency", "npu_chip_info_hbm_used_memory",
            "npu_chip_info_hbm_temperature", "npu_chip_info_hbm_bandwidth_utilization", "npu_chip_info_hbm_utilization"
        ]

        result = init_device_info(node_name, phy_id, card_id, device_id, fields)

        basic_info = HBMCollector._get_device_basic_info(logic_id, card_id, device_id)
        if "error_code" in basic_info:
            add_error(result, "mem_info", basic_info["error_code"], basic_info["error_message"])
            return result

        hbm_info = HBMCollector._get_device_hbm_info(logic_id, card_id, device_id)
        if "error_code" not in hbm_info:
            result.update({
                "npu_chip_info_hbm_total_memory": hbm_info["memory_size"],
                "npu_chip_info_hbm_frequency": hbm_info["frequency"],
                "npu_chip_info_hbm_used_memory": hbm_info["memory_usage"],
                "npu_chip_info_hbm_temperature": hbm_info["temperature"],
                "npu_chip_info_hbm_bandwidth_utilization": hbm_info["bandwidth_util_rate"],
            })
        else:
            add_error(result, "hbm_info", hbm_info["error_code"], hbm_info["error_message"])

        util_info = HBMCollector._get_device_hbm_utilization_rate(logic_id, card_id, device_id)
        if "error_code" not in util_info:
            result.update({
                "npu_chip_info_hbm_utilization": util_info["utilization_rate"]
            })
        else:
            add_error(result, "hbm_info", util_info["error_code"], util_info["error_message"])

        return result

    @staticmethod
    def _get_device_basic_info(logic_id: int, card_id: int, device_id: int):
        _, _, device_type, code, err = fetch_device_info_and_type(card_id, device_id)
        if err is not None:
            msg = f"Failed to resolve device type from logicID({logic_id}): {err}"
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": code, "error_message": msg}

        if device_type not in (ASCEND_910, ASCEND_910B, ASCEND_910A3, ASCEND_910A5):
            msg = f"LogicID({logic_id}): {device_type} does not have hbm module"
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {}

    @staticmethod
    def _get_device_hbm_info(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_hbm_info"):
            msg = "dcmi_get_device_hbm_info not available in shared object."
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        hbm_info_struct = DcmiHBMInfoStruct()
        ret = dcmi_instance.dcmi.dcmi_get_device_hbm_info(
            c_int(card_id), c_int(device_id), byref(hbm_info_struct)
        )
        if ret != 0:
            msg = f"dcmi_get_device_hbm_info failed for logicID({logic_id}), ret: {ret}."
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": ret, "error_message": msg}

        if hbm_info_struct.temp < 0:
            msg = f"Invalid HBM temperature for logicID({logic_id}), temp: {hbm_info_struct.temp}."
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {
            "memory_size": hbm_info_struct.memory_size,
            "frequency": hbm_info_struct.freq,
            "memory_usage": hbm_info_struct.memory_usage,
            "temperature": hbm_info_struct.temp,
            "bandwidth_util_rate": hbm_info_struct.bandwith_util_rate,
        }

    @staticmethod
    def _get_device_hbm_utilization_rate(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        rate, err = get_utilization_rate(card_id, device_id, HBM_DEVICE_TYPE_CODE)
        if err is not None:
            msg = f"Failed to get hbm utilization rate from logicID({logic_id}): {err}"
            logger.error("[HBMCollector] %s",msg)
            return {"error_code": rate, "error_message": msg}

        return {"utilization_rate": rate}


collector = HBMCollector()
registry.register(collector.name, collector, group="collector.npu")
