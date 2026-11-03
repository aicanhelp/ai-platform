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
This module defines some NPU constants.
"""

from collections import namedtuple

# pylint: disable=invalid-name

DCMI_LIBRARY_NAME = "libdcmi.so"

# HIAI_MAX_CARD_NUM max card number
HIAI_MAX_CARD_NUM = 64
# HIAI_MAX_DEVICE_NUM max device number
HIAI_MAX_DEVICE_NUM = 4

PERCENT = 100
LIMIT_SIZE = 1024*1024

# RET_INVALID_PARAMETER indicates that the input parameter or return value is invalid
RET_INVALID_VALUE = -1
# RET_NO_FUNCTION indicates that the required function is not found
RET_NO_FUNCTION = -2
RET_TIMEOUT = -3

DEVICE_NOT_READY_ERROR_CODE = -8012
CARD_DROP_FAULT_CODE = 0x40F84E00

NETWORK_INIT = 6
NETWORK_SUCCESS = 0

HBM_DEVICE_TYPE_CODE = 6
AI_CORE_DEVICE_TYPE_CODE = 2

MAX_CHIP_NAME_LEN = 32

INVALID_ID = 0xffffffff
INVALID_TEMPERATURE = -275

MAX_INT8 = 2**7 - 1
MAX_INT32 = 2**31 - 1
MAX_ERROR_CODE_COUNT = 128
BUF_SIZE = 256

A3_BOARD_IDS = {
    0xb0,  # A900A3SuperPodBin1BoardId
    0xb1,  # A900A3SuperPodBin2BoardId
    0xb2,  # A900A3SuperPodBin3BoardId
    0xb3,  # A800IA3BoardId
}
A5_BOARD_IDS = {
    0x28,  # A900A5SuperPodBin1BoardId
    0x29,  # A900A5SuperPodBin2BoardId
    0x2a,  # A900A5SuperPodBin3BoardId
    0x2b,  # A800A5ServerBin21BoardId
    0x2c,  # A800A5ServerBin22BoardId
    0x0c,  # A800A5ServerMultiModalBin22BoardId
    0x1a,  # A5300IBoardId
    0x1b,  # A5300IBoardId2
}

A300I_A2_Board_Id = 0x28
A300I_A2_GB64_Board_Id = 0x29

ASCEND_310P = "Ascend310P"
ASCEND_310B = "Ascend310B"
ASCEND_310 = "Ascend310"
ASCEND_910A5 = "Ascend910A5"
ASCEND_910A3 = "Ascend910A3"
ASCEND_910B = "Ascend910B"
ASCEND_910 = "Ascend910"
PATTERN_310P = r"^(310P\d{0,1}|I2\d{0,1})"
PATTERN_910A = r"^910"
PATTERN_910B = r"^(910B\d{1}|A2G\d{1})"

DeviceType = namedtuple('DeviceType', ['code', 'name'])

AICORE_CURRENT_FREQ = DeviceType(code=7, name="AICore Current")
AICORE_RATED_FREQ = DeviceType(code=9, name="AICore Rated")
