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
This module defines classes and methods for managing and interacting with the DCMI in NPU hardware.
"""
# pylint: disable=invalid-name

import ctypes
import logging
import os
import enum

from ctypes import c_int, c_uint, c_char

from common.npu_metrics.constants import MAX_CHIP_NAME_LEN

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


# pylint: disable=too-few-public-methods
class DcmiHBMInfoStruct(ctypes.Structure):
    """
    Represents the HBM (High Bandwidth Memory) information structure for a device.
    """
    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("freq", ctypes.c_uint),
        ("memory_usage", ctypes.c_ulonglong),
        ("temp", ctypes.c_int),
        ("bandwith_util_rate", ctypes.c_uint),
    ]


class DcmiChipInfoStruct(ctypes.Structure):
    """
    Represents the chip information structure for a device.
    """
    _fields_ = [
        ("chip_type", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("chip_name", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("chip_ver", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("aicore_cnt", ctypes.c_uint),
    ]


class DcmiChipInfoV2Struct(ctypes.Structure):
    """
    Represents an extended version of the chip information structure for a device.
    """
    _fields_ = [
        ("chip_type", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("chip_name", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("chip_ver", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
        ("aicore_cnt", ctypes.c_uint),
        ("npu_name", ctypes.c_ubyte * MAX_CHIP_NAME_LEN),
    ]


class DcmiBoardInfoStruct(ctypes.Structure):
    """
    Represents the board information structure for a device.
    """
    _fields_ = [
        ("board_id", ctypes.c_uint),
        ("pcb_id", ctypes.c_uint),
        ("bom_id", ctypes.c_uint),
        ("slot_id", ctypes.c_uint),
    ]


class DcmiMemoryInfoStruct(ctypes.Structure):
    """
    Represents the memory information structure for a device.
    """
    _fields_ = [
        ("memory_size", ctypes.c_ulonglong),
        ("memory_available", ctypes.c_ulonglong),
        ("freq", c_uint),
        ("hugepagesize", ctypes.c_ulong),
        ("hugepages_total", ctypes.c_ulong),
        ("hugepages_free", ctypes.c_ulong),
        ("utiliza", c_uint),
        ("reserve", ctypes.c_ubyte * 60),
    ]


class DcmiResetChannel(enum.IntEnum):
    """
    Enum representing the reset channels available for a device.
    """
    OUTBAND_CHANNEL = 0
    INBAND_CHANNEL = 1


class DcmiWrapper:
    """
    The DcmiWrapper class is responsible for managing the DCMI shared object (SO) library.
    """
    __instance__ = None

    def __new__(cls, lib_path: str):
        if cls.__instance__ is None:
            cls.__instance__ = super().__new__(cls)
            if not cls.__instance__._load_library(lib_path):
                cls.__instance__ = None
                return None
        return cls.__instance__

    def __init__(self, lib_path: str):
        if hasattr(self, 'dcmi'):
            return
        self.dcmi = None

        if lib_path:
            self._load_library(lib_path)

    def _load_library(self, lib_path: str) -> bool:
        """
        Loads the DCMI shared library and binds the necessary functions.
        """
        try:
            self.dcmi = ctypes.cdll.LoadLibrary(lib_path)
            logger.info("[DcmiWrapper] Successfully loaded library.")

            self.dcmi.dcmi_init.argtypes = []
            self.dcmi.dcmi_init.restype = ctypes.c_int

            ret = self.dcmi.dcmi_init()
            if ret != 0:
                logger.error("[DcmiWrapper] dcmi_init failed with ret=%s", ret)
                return False

            self._bind_functions()
            return True

        except OSError as e:
            logger.error("[DcmiWrapper] Failed to load the library: %s", str(e))
            return False
        except AttributeError as e:
            logger.error("[DcmiWrapper] Symbol not found in SO: %s", str(e))
            return False

    def _bind_functions(self):
        """
        Binds the necessary functions from the DCMI shared library.
        """
        if self.dcmi is None:
            return
        function_specs = {
            "dcmi_get_card_list": (
                [ctypes.POINTER(c_int), ctypes.POINTER(c_int), c_int], c_int
            ),
            "dcmi_get_device_num_in_card": (
                [c_int, ctypes.POINTER(c_int)], c_int
            ),
            "dcmi_get_device_logic_id": (
                [ctypes.POINTER(c_int), c_int, c_int], c_int
            ),
            "dcmi_get_device_chip_info": (
                [c_int, c_int, ctypes.POINTER(DcmiChipInfoStruct)], c_int
            ),
            "dcmi_get_device_chip_info_v2": (
                [c_int, c_int, ctypes.POINTER(DcmiChipInfoV2Struct)], c_int
            ),
            "dcmi_get_device_board_info": (
                [c_int, c_int, ctypes.POINTER(DcmiBoardInfoStruct)], c_int
            ),
            "dcmi_get_device_hbm_info": (
                [c_int, c_int, ctypes.POINTER(DcmiHBMInfoStruct)], c_int
            ),
            "dcmi_get_device_utilization_rate": (
                [c_int, c_int, c_int, ctypes.POINTER(c_uint)], c_int
            ),
            "dcmi_get_device_memory_info_v3": (
                [c_int, c_int, ctypes.POINTER(DcmiMemoryInfoStruct)], c_int
            ),
            "dcmi_get_device_temperature": (
                [c_int, c_int, ctypes.POINTER(c_int)], c_int
            ),
            "dcmi_get_device_frequency": (
                [c_int, c_int, c_int, ctypes.POINTER(c_uint)], c_int
            ),
            "dcmi_get_device_health": (
                [c_int, c_int, ctypes.POINTER(c_uint)], c_int
            ),
            "dcmi_get_device_errorcode_v2": (
                [c_int, c_int, ctypes.POINTER(c_int), ctypes.POINTER(c_uint), c_uint], c_int
            ),
            "dcmi_get_device_errorcode_string": (
                [c_int, c_int, c_uint, ctypes.POINTER(c_char), c_uint], c_int
            ),
            "dcmi_get_device_network_health": (
                [c_int, c_int, ctypes.POINTER(c_uint)], c_int
            ),
            "dcmi_get_device_phyid_from_logicid": (
                [c_uint, ctypes.POINTER(c_uint)], c_int
            ),
            "dcmi_set_device_reset": (
                [c_int, c_int, c_int], c_int
            )

        }
        for func_name, (argtypes, restype) in function_specs.items():
            try:
                func = getattr(self.dcmi, func_name)
                func.argtypes = argtypes
                func.restype = restype
            except AttributeError:
                logger.error("[DcmiWrapper] Function %s not found in shared object.", func_name)
                continue
