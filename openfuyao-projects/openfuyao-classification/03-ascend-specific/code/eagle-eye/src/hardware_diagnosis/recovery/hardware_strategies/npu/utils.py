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
This module queries DCMI error messages and performs device resets.
"""

from typing import Union
from ctypes import c_int, c_uint, c_char

from common.npu_metrics.constants import RET_NO_FUNCTION, BUF_SIZE
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.dcmi.dcmi_wrapper import DcmiResetChannel


def get_device_error_code_string(logic_id: int,
                                 card_id: int,
                                 device_id: int,
                                 error_code: int) -> tuple[Union[str, None], Union[int, None], Union[Exception, None]]:
    """
    Queries the DCMI to retrieve the error message associated with a device error code.
    """
    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance.dcmi is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_errorcode_string"):
        return None, RET_NO_FUNCTION, RuntimeError("dcmi_get_device_board_info "
                                                   "function not found in dcmi shared object")

    buf = (c_char * BUF_SIZE)()
    ret = dcmi_instance.dcmi.dcmi_get_device_errorcode_string(
        c_int(card_id), c_int(device_id), c_uint(error_code), buf, c_uint(BUF_SIZE)
    )

    if ret != 0:
        return None, ret, RuntimeError("logicID{%s}, cardID{%s}, deviceID{%s}, ret: {%s}",
                                       logic_id, card_id, device_id, ret)

    msg = bytes(buf).split(b"\x00", 1)[0].decode(errors="ignore")

    return msg, None, None


def set_device_reset(logic_id: int,
                     card_id: int,
                     device_id: int) -> tuple[Union[int, None], Union[Exception, None]]:
    """
    Resets a device by triggering the DCMI reset function.
    """
    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance.dcmi is None or not hasattr(dcmi_instance.dcmi, "dcmi_set_device_reset"):
        return RET_NO_FUNCTION, RuntimeError("dcmi_set_device_reset "
                                             "function not found in dcmi shared object")

    ret = dcmi_instance.dcmi.dcmi_set_device_reset(
        c_int(card_id),
        c_int(device_id),
        c_int(DcmiResetChannel.INBAND_CHANNEL)
    )

    if ret != 0:
        return ret, RuntimeError("logicID{%s}, cardID{%s}, deviceID{%s}, ret: {%s}",
                                 logic_id, card_id, device_id, ret)

    return ret, None
