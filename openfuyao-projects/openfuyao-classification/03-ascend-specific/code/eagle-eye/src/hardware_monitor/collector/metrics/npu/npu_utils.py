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
This module defines some functions to get npu info.
"""

from ctypes import c_int, c_uint, byref
from typing import List, Any

from common.npu_metrics.constants import (PERCENT, RET_INVALID_VALUE, RET_NO_FUNCTION, MAX_INT32)
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.utils import is_valid_card_and_device_id
from hardware_monitor.base.utils import init_metric_fields


def is_valid_utilization_rate(num: int) -> bool:
    """
    Checks if the utilization rate is valid.
    """
    return 0 <= num <= PERCENT


def is_greater_than_or_equal_int32(num: int) -> bool:
    """
    Checks if the given number is greater than or equal to the maximum value of a 32-bit integer.
    """
    return num >= MAX_INT32


def init_device_info(
        node_name: str,
        phy_id: int,
        card_id: int,
        device_id: int,
        fields: List[str],
) -> dict[str, Any]:
    """
    Initializes device information with a set of fields.
    """
    result = init_metric_fields(fields)
    result.update({
        "node_name": node_name,
        "phy_id": phy_id,
        "card_id": card_id,
        "device_id": device_id
    })
    return result


def get_utilization_rate(card_id: int, device_id: int,
                         device_type_code: int):
    """
    Retrieves the utilization rate for a given card and device ID.
    """
    if not is_valid_card_and_device_id(card_id, device_id):
        return RET_INVALID_VALUE, ValueError(
            f"invalid cardID({card_id}) or deviceID({device_id})"
        )

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_utilization_rate"):
        return RET_NO_FUNCTION, RuntimeError("dcmi_get_device_utilization_rate function not found "
                                             "in dcmi shared object")

    rate = c_uint(0)
    ret = dcmi_instance.dcmi.dcmi_get_device_utilization_rate(
        c_int(card_id), c_int(device_id), c_int(device_type_code), byref(rate)
    )
    if ret != 0:
        return ret, RuntimeError(f"ret: {ret}")

    rate_value = rate.value
    if not is_valid_utilization_rate(rate_value):
        return (RET_INVALID_VALUE,
                ValueError(f"invalid utilization rate, cardID({card_id}), deviceID({device_id})"))

    return rate_value, None
