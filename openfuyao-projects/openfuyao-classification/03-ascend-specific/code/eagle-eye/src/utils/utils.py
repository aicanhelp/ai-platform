# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import logging

from common.npu_device_info.npu_discovery import get_lib_from_ld_cmd
from common.npu_metrics.constants import DCMI_LIBRARY_NAME
from common.npu_metrics.dcmi.dcmi import DcmiManager

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


def detect_devices():
    """
    Detects the available hardware devices (NPU and GPU)
    by checking their respective initialization.
    """
    devices = {"npu": check_npu_devices_and_initialize(),
               "gpu": check_gpu_devices_and_initialize()}

    return devices


def check_npu_devices_and_initialize():
    """
    Checks if NPU devices are available and attempts to initialize them using the DCMI library.
    """
    lib_path, error = get_lib_from_ld_cmd(DCMI_LIBRARY_NAME)
    if lib_path:
        if DcmiManager.initialize_dcmi(lib_path):
            return True
        logger.error("Failed to initialize dcmi.")
        return False
    logger.info(error)
    return False


def check_gpu_devices_and_initialize():
    """
    Checks if GPU devices are available and initializes them (currently a placeholder).
    """
    return False
