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
This module defines NPUCollector class.
"""

import logging
import os
import threading
import queue
from ctypes import byref, c_int, c_uint
from typing import Any

from common.base.base_metrics_collector import BaseCollector
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.constants import (RET_INVALID_VALUE, RET_NO_FUNCTION, RET_TIMEOUT,
                                          AI_CORE_DEVICE_TYPE_CODE, INVALID_TEMPERATURE,
                                          MAX_ERROR_CODE_COUNT, DEVICE_NOT_READY_ERROR_CODE,
                                          CARD_DROP_FAULT_CODE, MAX_INT8, NETWORK_INIT,
                                          NETWORK_SUCCESS, AICORE_CURRENT_FREQ, AICORE_RATED_FREQ)
from common.npu_metrics.utils import fetch_device_info_and_type, is_training_card
from hardware_monitor.base.registry import registry
from hardware_monitor.base.utils import add_error, enrich_result_with_container_info
from hardware_monitor.collector.metrics.npu.npu_utils import (init_device_info, get_utilization_rate,
                                                              is_greater_than_or_equal_int32)

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class NPUCollector(BaseCollector):
    """
    This collector collects information from NPU.
    """

    @property
    def name(self):
        return "npu"

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
            info = self._get_device_all_npu_info(node_name, logic_id, phy_id, card_id, device_id)
            results[str(logic_id)] = enrich_result_with_container_info(info, phy_id, self.cache)
        return results

    @staticmethod
    def _get_device_all_npu_info(node_name: str, logic_id: int, phy_id: int,
                                 card_id: int, device_id: int) -> dict[str, Any]:
        fields = [
            "node_name", "phy_id", "card_id", "device_id",
            "namespace", "pod_name", "container_name",
            "npu_chip_info_aicore_utilization_rate", "npu_chip_info_temperature",
            "npu_chip_info_aicore_current_freq", "npu_chip_info_aicore_rated_freq", "npu_chip_info_health_status",
            "npu_chip_info_error_code", "npu_chip_info_network_status"
        ]

        result = init_device_info(node_name, phy_id, card_id, device_id, fields)

        rate = NPUCollector._get_device_utilization_rate(logic_id, card_id, device_id)
        if "error_code" not in rate:
            result.update({"npu_chip_info_aicore_utilization_rate": rate["utilization_rate"]})
        else:
            add_error(result, "npu_chip_info", rate["error_code"], rate["error_message"])

        temp = NPUCollector._get_device_temperature(logic_id, card_id, device_id)
        if "error_code" not in temp:
            result.update({"npu_chip_info_temperature": temp["temperature"]})
        else:
            add_error(result, "npu_chip_info", temp["error_code"], temp["error_message"])

        current_freq = NPUCollector._get_device_frequency(logic_id, card_id, device_id, AICORE_CURRENT_FREQ)
        if "error_code" not in current_freq:
            result.update({"npu_chip_info_aicore_current_freq": current_freq["frequency"]})
        else:
            add_error(result, "npu_chip_info", current_freq["error_code"],
                      current_freq["error_message"])

        rated_freq = NPUCollector._get_device_frequency(logic_id, card_id, device_id, AICORE_RATED_FREQ)
        if "error_code" not in rated_freq:
            result.update({"npu_chip_info_aicore_rated_freq": rated_freq["frequency"]})
        else:
            add_error(result, "npu_chip_info", rated_freq["error_code"],
                      rated_freq["error_message"])

        health = NPUCollector._get_device_health(logic_id, card_id, device_id)
        if "error_code" not in health:
            result.update({"npu_chip_info_health_status": health["health"]})
        else:
            add_error(result, "npu_chip_info", health["error_code"], health["error_message"])

        error = NPUCollector._get_device_all_error_code(logic_id, card_id, device_id)
        if "error_code" not in error:
            result.update({"npu_chip_info_error_code": error["error_codes"]})
        else:
            add_error(result, "npu_chip_info", error["error_code"], error["error_message"])

        network_health = NPUCollector._get_device_network_status(logic_id, card_id, device_id)
        if "error_code" not in network_health:
            if network_health["health"] in (NETWORK_INIT, NETWORK_SUCCESS):
                network_health["health"] = 0
            else:
                network_health["health"] = 1
            result.update({"npu_chip_info_network_status": network_health["health"]})
        else:
            network_health["health"] = -1
            add_error(result, "npu_chip_info", network_health["error_code"], network_health["error_message"])

        return result

    @staticmethod
    def _get_device_utilization_rate(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        rate, err = get_utilization_rate(card_id, device_id, AI_CORE_DEVICE_TYPE_CODE)
        if err is not None:
            msg = f"Failed to get chip info from logicID({logic_id}): {err}"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": rate, "error_message": msg}

        return {"utilization_rate": rate}

    @staticmethod
    def _get_device_temperature(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_temperature"):
            msg = "dcmi_get_device_temperature not available in shared object."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        temp = c_int()
        ret = dcmi_instance.dcmi.dcmi_get_device_temperature(c_int(card_id), c_int(device_id), byref(temp))

        if ret != 0:
            msg = f"Failed to get temperature for logicID({logic_id}), ret: {ret}."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": ret, "error_message": msg}

        parsed_temp = temp.value
        if parsed_temp < INVALID_TEMPERATURE:
            msg = f"Invalid temperature value for logicID({logic_id})"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {"temperature": parsed_temp}

    @staticmethod
    def _get_device_frequency(logic_id: int, card_id: int, device_id: int, dev_type) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_frequency"):
            msg = "dcmi_get_device_frequency not available in shared object."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        c_frequency = c_uint(0)
        ret = dcmi_instance.dcmi.dcmi_get_device_frequency(
            c_int(card_id), c_int(device_id), c_int(dev_type.code), byref(c_frequency)
        )

        if ret != 0:
            msg = (f"Failed to get frequency for logicID({logic_id}), "
                   f"freq_type(name={dev_type.name}, code={dev_type.code}), ret: {ret}.")
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": ret, "error_message": msg}

        freq_val = int(c_frequency.value)
        if is_greater_than_or_equal_int32(freq_val) or freq_val < 0:
            msg = (f"frequency value out of range [0, int32), card_id({card_id}) and device_id({device_id}), "
                   f"frequency (name={dev_type.name}, code={dev_type.code}): {freq_val}")
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {"frequency": freq_val}

    @staticmethod
    def _get_device_health(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_health"):
            msg = "dcmi_get_device_health not available in shared object."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        health = c_uint()
        ret = dcmi_instance.dcmi.dcmi_get_device_health(
            c_int(card_id), c_int(device_id), byref(health)
        )

        if ret != 0:
            msg = f"Failed to get device health for logicID({logic_id}), ret: {ret}."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": ret, "error_message": msg}

        if is_greater_than_or_equal_int32(health.value):
            msg = f"get wrong health state, device (cardID: {card_id}, deviceID: {device_id}), health: {health.value}"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {"health": 1 if health.value != 0 else 0}

    @staticmethod
    def _get_device_all_error_code(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if (dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_errorcode_v2") or
                not hasattr(dcmi_instance.dcmi, "dcmi_get_device_health")):
            msg = "dcmi_get_device_errorcode_v2 / dcmi_get_device_health not available in shared object."
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        err_count = c_int()
        err_code_array = (c_uint * MAX_ERROR_CODE_COUNT)()
        ret = dcmi_instance.dcmi.dcmi_get_device_errorcode_v2(
            c_int(card_id), c_int(device_id), byref(err_count),
            err_code_array, c_uint(MAX_ERROR_CODE_COUNT)
        )

        health = c_uint()
        health_ret = dcmi_instance.dcmi.dcmi_get_device_health(
            c_int(card_id), c_int(device_id), byref(health)
        )

        if ret != 0 and health_ret != DEVICE_NOT_READY_ERROR_CODE:
            msg = (f"Failed to obtain the device error code for logicID({logic_id}), ret: {ret}, "
                   f"health ret: {health_ret}.")
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": ret or health_ret, "error_message": msg}

        error_codes = [int(code) for code in err_code_array if int(code) != 0]

        if health_ret == DEVICE_NOT_READY_ERROR_CODE:
            msg = "maybe a card drop"
            logger.warning("[NPUCollector] %s", msg)
            err_count.value += 1
            error_codes.append(int(CARD_DROP_FAULT_CODE))

        if err_count.value < 0 or err_count.value > MAX_ERROR_CODE_COUNT:
            msg = f"invalid error code count for logicID({logic_id})"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_INVALID_VALUE, "error_message": msg}

        return {
            "error_count": int(err_count.value),
            "error_codes": error_codes
        }

    @staticmethod
    def _should_probe_network(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        _, board_info, device_type, code, err = fetch_device_info_and_type(card_id, device_id)
        if err is not None:
            msg = f"Failed to resolve device type from logicID({logic_id}): {err}"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": code, "error_message": msg}

        board_id = board_info.get("board_id")
        if not is_training_card(device_type, board_id):
            return {"health": -1}

        return {"health": None}

    @staticmethod
    def _probe_network_health(card_id: int, device_id: int, timeout_sec: float = 1.0) -> dict[str, Any]:
        dcmi_instance = DcmiManager.get_dcmi_instance()
        if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_network_health"):
            msg = "dcmi_get_device_network_health not available in shared object"
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": RET_NO_FUNCTION, "error_message": msg}

        q: "queue.Queue[tuple[int|None, int|None, Exception|None]]" = queue.Queue(maxsize=1)

        def worker():
            try:
                health = c_uint()
                ret = dcmi_instance.dcmi.dcmi_get_device_network_health(c_int(card_id),
                                                                        c_int(device_id),
                                                                        byref(health))
                if ret != 0:
                    q.put((None, ret, RuntimeError(f"errorcode: {ret}, cardID={card_id}, deviceID{device_id}")))
                    return
                value = int(health.value)
                if value > MAX_INT8:
                    q.put((None, RET_INVALID_VALUE, ValueError(f"invalid health_code={value}")))
                    return
                q.put((value, None, None))
            except Exception as e:  # pylint: disable=broad-except
                q.put((None, RET_INVALID_VALUE, e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        try:
            health_code, ret_code, err = q.get(timeout=timeout_sec)
        except queue.Empty:
            msg = (f"dcmi_get_device_network_health timeout after {timeout_sec}s "
                   f"(card_id={card_id}, device_id={device_id})")
            logger.warning("[NPUCollector] %s", msg)
            return {"error_code": RET_TIMEOUT, "error_message": msg}

        if err is not None:
            msg = (f"dcmi_get_device_network_health failed (card_id={card_id}, device_id={device_id}), "
                   f"ret_code={ret_code}, err={err}")
            logger.error("[NPUCollector] %s", msg)
            return {"error_code": ret_code or RET_INVALID_VALUE, "error_message": msg}

        return {"health": int(health_code)}

    @staticmethod
    def _get_device_network_status(logic_id: int, card_id: int, device_id: int) -> dict[str, Any]:
        gate = NPUCollector._should_probe_network(logic_id, card_id, device_id)
        if "error_code" in gate:
            return gate
        if gate["health"] == -1:
            return {"health": -1}

        return NPUCollector._probe_network_health(card_id, device_id, timeout_sec=1.0)


collector = NPUCollector()
registry.register(collector.name, collector, group="collector.npu")
