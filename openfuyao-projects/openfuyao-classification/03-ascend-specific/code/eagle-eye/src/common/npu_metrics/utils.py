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

import re

from ctypes import c_int, c_uint, byref, Array
from typing import Tuple, List, Union

from common.npu_metrics.constants import (HIAI_MAX_CARD_NUM, HIAI_MAX_DEVICE_NUM,
                                          RET_INVALID_VALUE, RET_NO_FUNCTION,
                                          INVALID_ID, A3_BOARD_IDS, A5_BOARD_IDS,
                                          A300I_A2_Board_Id, ASCEND_910A3, ASCEND_910A5,
                                          ASCEND_310, ASCEND_310B, ASCEND_310P,
                                          ASCEND_910, ASCEND_910B,
                                          PATTERN_910A, PATTERN_910B, PATTERN_310P)
from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.dcmi.dcmi_wrapper import DcmiChipInfoV2Struct, DcmiChipInfoStruct, \
    DcmiBoardInfoStruct


def is_valid_logic_id_or_phy_id(logic_id_or_phy_id: int) -> bool:
    """
    Checks if the given logic ID or physical ID is valid.
    """
    return 0 <= logic_id_or_phy_id < HIAI_MAX_CARD_NUM * HIAI_MAX_DEVICE_NUM


def is_valid_card_id(card_id: int) -> bool:
    """
    Checks if the given card ID is valid.
    """
    return 0 <= card_id < HIAI_MAX_CARD_NUM


def is_valid_device_id(device_id: int) -> bool:
    """
    Checks if the given device ID is valid.
    """
    return 0 <= device_id < HIAI_MAX_DEVICE_NUM


def is_valid_device_num(num: int) -> bool:
    """
    Checks if the number of devices is valid.
    """
    return 0 < num <= HIAI_MAX_DEVICE_NUM


def is_valid_card_and_device_id(card_id: int, device_id: int) -> bool:
    """
    Checks if the given card ID and device ID are both valid.
    """
    return is_valid_card_id(card_id) and is_valid_device_id(device_id)


def get_device_logic_id(card_id: int, device_id: int) -> Tuple[int, Union[Exception, None]]:
    """
    Retrieves the device logic ID for a given card and device ID.
    """
    if not is_valid_card_and_device_id(card_id, device_id):
        return RET_INVALID_VALUE, ValueError(
            f"invalid cardID({card_id}) or deviceID({device_id})"
        )

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_logic_id"):
        return (RET_NO_FUNCTION,
                RuntimeError(
                    "dcmi_get_device_logic_id function not found in dcmi shared object")
                )

    logic_id = c_int(0)
    ret = dcmi_instance.dcmi.dcmi_get_device_logic_id(byref(logic_id), c_int(card_id), c_int(device_id))
    if ret != 0:
        return ret, RuntimeError(f"ret: {ret}")

    logic_id_value = logic_id.value
    if not is_valid_logic_id_or_phy_id(logic_id_value):
        return (RET_INVALID_VALUE,
                ValueError(f"invalid logicID, cardID({card_id}), deviceID({device_id})"))

    return logic_id_value, None


def get_device_num_in_card(card_id: int) -> Tuple[int, Union[Exception, None]]:
    """
    Retrieves the number of devices present on a given card.
    """
    if not is_valid_card_id(card_id):
        return RET_INVALID_VALUE, ValueError(f"invalid cardID({card_id})")

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_num_in_card"):
        return (RET_NO_FUNCTION,
                RuntimeError(
                    "dcmi_get_device_num_in_card function not found in dcmi shared object")
                )

    device_num = c_int(0)
    ret = dcmi_instance.dcmi.dcmi_get_device_num_in_card(c_int(card_id), byref(device_num))
    if ret != 0:
        return ret, RuntimeError(f"ret: {ret}")

    device_num_value = device_num.value
    if not is_valid_device_num(device_num_value):
        return (RET_INVALID_VALUE,
                ValueError(f"invalid device num, cardID({card_id}))"))

    return device_num_value, None


def get_card_list() -> Tuple[int, List[int], List[int], Union[Exception, None]]:
    """
    Retrieves a list of card IDs present on the system along with invalid card IDs.
    """
    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_card_list"):
        return (RET_NO_FUNCTION, [], [],
                RuntimeError("dcmi_get_card_list function not found in dcmi shared object"))

    num = c_int(0)
    buf = (c_int * HIAI_MAX_CARD_NUM)()
    ret = dcmi_instance.dcmi.dcmi_get_card_list(byref(num), buf, c_int(HIAI_MAX_CARD_NUM))
    if ret != 0:
        return ret, [], [], RuntimeError(f"ret: {ret}")

    n = int(num.value)
    if n <= 0 or n > HIAI_MAX_CARD_NUM:
        return RET_INVALID_VALUE, [], [], ValueError(
            f"invalid card quantity: card quantity is {n}"
        )

    ids = []
    invalid_ids = []
    for i in range(n):
        cid = int(buf[i])
        if cid < 0:
            invalid_ids.append(cid)
            continue
        ids.append(cid)

    if not ids:
        return (RET_INVALID_VALUE, ids, invalid_ids,
                ValueError("all returned card IDs are invalid (<0)"))
    return n, ids, invalid_ids, None


def get_phy_id(logic_id: int):
    """
    Retrieves the physical ID for a given logic ID.
    """
    if not is_valid_logic_id_or_phy_id(logic_id):
        return RET_INVALID_VALUE, ValueError("input invalid logicID")

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_phyid_from_logicid"):
        return RET_NO_FUNCTION, RuntimeError("dcmi_get_device_phyid_from_logicid "
                                             "function not found in dcmi shared object")

    phy_id = c_uint(0)
    ret = dcmi_instance.dcmi.dcmi_get_device_phyid_from_logicid(c_uint(logic_id), byref(phy_id))
    if ret != 0:
        return ret, RuntimeError(f"ret: {ret}")

    phy_value = phy_id.value
    if not is_valid_logic_id_or_phy_id(phy_value):
        return RET_INVALID_VALUE, ValueError(f"invalid phyID, logicID({logic_id})")

    return phy_value, None


def uchar_array_to_str(arr: Array) -> str:
    """
    Converts an unsigned char array to a string, stopping at the first null byte.
    """
    return bytes(arr).split(b"\x00", 1)[0].decode(errors="ignore")


def is_valid_chip_info(chip: dict) -> bool:
    """
    Validates if the chip information dictionary contains valid data.
    """
    return (
            bool(chip.get("chip_name")) or
            bool(chip.get("chip_type")) or
            bool(chip.get("chip_version"))
    )


def is_valid_board_info(board: dict) -> bool:
    """
    Validates if the board information dictionary contains valid data.
    """
    return (
            board.get("board_id", INVALID_ID) != INVALID_ID or
            board.get("pcb_id", INVALID_ID) != INVALID_ID or
            board.get("bom_id", INVALID_ID) != INVALID_ID or
            board.get("slot_id", INVALID_ID) != INVALID_ID
    )


def is_910a3_chip(board_id: int) -> bool:
    """
    Checks if the given board ID corresponds to an ASCEND 910A3 chip.
    """
    return board_id in A3_BOARD_IDS


def is_910a5_chip(board_id: int) -> bool:
    """
    Checks if the given board ID corresponds to an ASCEND 910A5 chip.
    """
    return board_id in A5_BOARD_IDS


def get_device_type_by_chip_name(chip_name: str) -> str:
    """
    Determines the device type based on the chip name.
    """
    reg_910a = re.compile(PATTERN_910A)
    reg_910b = re.compile(PATTERN_910B)
    reg_310p = re.compile(PATTERN_310P)
    if reg_310p.search(chip_name):
        return ASCEND_310P
    if "310B" in chip_name:
        return ASCEND_310B
    if "310" in chip_name:
        return ASCEND_310
    if reg_910b.search(chip_name):
        return ASCEND_910B
    if reg_910a.search(chip_name):
        return ASCEND_910
    return ""


def get_device_type(chip_name: str, board_id: int) -> str:
    """
    Determines the device type based on the chip name and board ID.
    """
    if is_910a3_chip(board_id):
        return ASCEND_910A3
    if is_910a5_chip(board_id):
        return ASCEND_910A5
    return get_device_type_by_chip_name(chip_name)


def is_training_card(device_type: str, board_id: int) -> bool:
    """
    Checks if the device is a training card based on the device type and board ID.
    """
    if device_type and device_type.startswith(ASCEND_310):
        return False
    if device_type == ASCEND_910B and board_id == A300I_A2_Board_Id:
        return False
    return True


def get_chip_info(card_id: int, device_id: int):
    """
    Retrieves information about the chip for a given card and device ID.
    """
    if not is_valid_card_and_device_id(card_id, device_id):
        return {}, RET_INVALID_VALUE, ValueError(
            f"invalid cardID({card_id}) or deviceID({device_id})"
        )

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or (not hasattr(dcmi_instance.dcmi, "dcmi_get_device_chip_info_v2")
                                 and not hasattr(dcmi_instance.dcmi, "dcmi_get_device_chip_info")):
        return {}, RET_NO_FUNCTION, RuntimeError("dcmi_get_device_chip_info_v2 / "
                                                 "dcmi_get_device_chip_info function not found "
                                                 "in dcmi shared object")

    chip_info_v2 = DcmiChipInfoV2Struct()
    try:
        ret = dcmi_instance.dcmi.dcmi_get_device_chip_info_v2(
            c_int(card_id), c_int(device_id), byref(chip_info_v2)
        )
    except AttributeError:
        ret = RET_NO_FUNCTION
    if ret != 0:
        chip_info_old = DcmiChipInfoStruct()
        ret_old = dcmi_instance.dcmi.dcmi_get_device_chip_info(
            c_int(card_id), c_int(device_id), byref(chip_info_old)
        )
        if ret_old != 0:
            return ({}, ret_old,
                    RuntimeError(f"cardID({card_id}), deviceID({device_id}), ret: {ret_old}"))
        chip_info = {
            "chip_type": uchar_array_to_str(chip_info_old.chip_type),
            "chip_name": uchar_array_to_str(chip_info_old.chip_name),
            "chip_version": uchar_array_to_str(chip_info_old.chip_ver),
            "aicore_cnt": int(chip_info_old.aicore_cnt)
        }
        if is_valid_chip_info(chip_info):
            return chip_info, None, None
        return ({}, RET_INVALID_VALUE,
                ValueError(f"chip info is empty, cardID({card_id}), deviceID({device_id})"))
    chip_info = {
        "chip_type": uchar_array_to_str(chip_info_v2.chip_type),
        "chip_name": uchar_array_to_str(chip_info_v2.chip_name),
        "chip_version": uchar_array_to_str(chip_info_v2.chip_ver),
        "aicore_cnt": int(chip_info_v2.aicore_cnt),
        "npu_name": uchar_array_to_str(chip_info_v2.npu_name)
    }
    if is_valid_chip_info(chip_info):
        return chip_info, None, None
    return ({}, RET_INVALID_VALUE,
            ValueError(f"chip info is empty, cardID({card_id}), deviceID({device_id})"))


def get_board_info(card_id: int, device_id: int):
    """
    Retrieves the board information for a given card and device ID.
    """
    if not is_valid_card_and_device_id(card_id, device_id):
        return {}, RET_INVALID_VALUE, ValueError(
            f"invalid cardID({card_id}) or deviceID({device_id})"
        )

    dcmi_instance = DcmiManager.get_dcmi_instance()
    if dcmi_instance is None or not hasattr(dcmi_instance.dcmi, "dcmi_get_device_board_info"):
        return {}, RET_NO_FUNCTION, RuntimeError("dcmi_get_device_board_info "
                                                 "function not found in dcmi shared object")

    board_info = DcmiBoardInfoStruct()
    ret = dcmi_instance.dcmi.dcmi_get_device_board_info(
        c_int(card_id), c_int(device_id), byref(board_info)
    )
    if ret != 0:
        return {}, ret, RuntimeError(f"ret: {ret}")

    board_info_result = {
        "board_id": int(board_info.board_id),
        "pcb_id": int(board_info.pcb_id),
        "bom_id": int(board_info.bom_id),
        "slot_id": int(board_info.slot_id)
    }

    if is_valid_board_info(board_info_result):
        return board_info_result, None, None
    return {}, RET_INVALID_VALUE, ValueError(f"invalid board info, cardID({card_id}), deviceID({device_id})")


def fetch_device_info_and_type(card_id: int, device_id: int):
    """
    Retrieves the chip info, board info, and device type for a given card and device ID.
    """
    chip_info, code, err = get_chip_info(card_id, device_id)
    if err is not None:
        return chip_info, {}, None, code, err

    board_info, code, err = get_board_info(card_id, device_id)
    if err is not None:
        return chip_info, board_info, None, code, err

    device_type = get_device_type(
        chip_info.get("chip_name"),
        board_info.get("board_id")
    )
    if not device_type:
        return (chip_info, board_info, None, RET_INVALID_VALUE,
                ValueError(f"device type is empty, cardID({card_id}), deviceID({device_id})"))

    return chip_info, board_info, device_type, None, None
