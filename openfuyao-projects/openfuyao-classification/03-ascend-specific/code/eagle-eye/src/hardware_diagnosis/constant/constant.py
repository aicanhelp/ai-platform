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
This module defines DCMIBackCode, RecoveryAction classes and some constant.
"""
# pylint: disable=invalid-name
from enum import Enum
from typing import Union, Dict, Any, List


class DCMIBackCode(Enum):
    """
    This class lists the function back error codes.
    """

    DCMI_ERR_CODE_INVALID_PARAMETER = "-8001"
    DCMI_ERR_CODE_OPER_NOT_PERMITTED = "-8002"
    DCMI_ERR_CODE_MEM_OPERATE_FAIL = "-8003"
    DCMI_ERR_CODE_SECURE_FUN_FAIL = "-8004"
    DCMI_ERR_CODE_INNER_ERR = "-8005"
    DCMI_ERR_CODE_TIME_OUT = "-8006"
    DCMI_ERR_CODE_INVALID_DEVICE_ID = "-8007"
    DCMI_ERR_CODE_DEVICE_NOT_EXIST = "-8008"
    DCMI_ERR_CODE_IOCTL_FAIL = "-8009"
    DCMI_ERR_CODE_SEND_MSG_FAIL = "-8010"
    DCMI_ERR_CODE_RECV_MSG_FAIL = "-8011"
    DCMI_ERR_CODE_NOT_READY = "-8012"
    DCMI_ERR_CODE_NOT_SUPPORT_IN_CONTAINER = "-8013"
    DCMI_ERR_CODE_RESET_FAIL = "-8015"
    DCMI_ERR_CODE_ABORT_OPERATE = "-8016"
    DCMI_ERR_CODE_IS_UPGRADING = "-8017"
    DCMI_ERR_CODE_RESOURCE_OCCUPIED = "-8020"
    DCMI_ERR_CODE_CONFIG_INFO_NOT_EXIST = "-8023"
    DCMI_ERR_CODE_NOT_SUPPORT = "-8055"


class RecoveryAction(Enum):
    """
    This class lists the recovery actions.
    """
    RESTART_HOST = "Restart Host"
    RESET_NPU_DEVICE = "Restart NPU device soc"
    CONTACT_SUPPORT = "Contact support"
    IGNORE = "Ignore"
    UPDATE_TLS = "Update TLS certificate of device"
    CHECK_NETWORK = "Check link status of network port"
    LOWER_TEMPERATURE = "Lower temperature"
    CHECK_NPU_STATUS = "check NPU status"
    RESET_GPU = "Reset GPU"

    RESTART_WORKER = "Restart worker"
    RELAUNCH_WORKER = "Relaunch worker"


ERROR_CODE_MAP: Dict[Union[int, str], Any] = {
    hex(0x8C2FA001): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x812E4E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x80E20207): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x80E21008): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x80E24E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x80FA4E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C03A000): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C044E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C064E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C084E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C0C4E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C0E4E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C104E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C124E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C17A005): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C19A005): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C1DA005): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C1FA006): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],
    hex(0x8C204E00): [RecoveryAction.RESET_NPU_DEVICE, RecoveryAction.RESTART_WORKER],

    hex(0x80818200): [RecoveryAction.UPDATE_TLS, RecoveryAction.RESTART_WORKER],
    hex(0x80818201): [RecoveryAction.UPDATE_TLS, RecoveryAction.RESTART_WORKER],

    hex(0x8C464E00): RecoveryAction.IGNORE,
    hex(0x80E21007): RecoveryAction.IGNORE,
    hex(0x8C0A4E00): RecoveryAction.IGNORE,
    hex(0x8C2FA009): RecoveryAction.IGNORE,

    hex(0x8C1F8608): [RecoveryAction.CHECK_NETWORK],
    hex(0x4C1F8608): [RecoveryAction.CHECK_NETWORK],

    hex(0xFFFFFFFF): [RecoveryAction.CONTACT_SUPPORT, RecoveryAction.RELAUNCH_WORKER],

    hex(100): [RecoveryAction.LOWER_TEMPERATURE],
    hex(101): [RecoveryAction.CHECK_NPU_STATUS],
}

XID_MAP: Dict[str, Any] = {
    "140": {
        "describe": "ECC unrecovered error,",
        "need_further": True,
        "suggestion": [RecoveryAction.RESET_GPU, RecoveryAction.RESTART_WORKER],
        "severity": 1
    },
    "60": {
        "describe": "GPU driver error: Video processor exception",
        "need_further": False,
        "suggestion": [RecoveryAction.RESET_GPU, RecoveryAction.RESTART_WORKER],
        "severity": 3
    },
}

HBM_TEMPERATURE_THRESHOLD = 90
NPU_CHIP_TEMPERATURE_THRESHOLD = 105
DEFAULT_STAGES: List[str] = ["xid_basic_analyzer"]
