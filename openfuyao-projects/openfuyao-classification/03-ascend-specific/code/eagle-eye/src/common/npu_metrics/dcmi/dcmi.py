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
This module defines the `DcmiManager` class, which is responsible for managing the DCMI instance.
"""

import logging
import os

from common.npu_metrics.dcmi.dcmi_wrapper import DcmiWrapper

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class DcmiManager:
    """
    The DcmiManager class is responsible for managing the lifecycle of the DCMI instance.
    """
    _dcmi_instance = None

    @classmethod
    def initialize_dcmi(cls, lib_path: str) -> bool:
        """
        Initializes the DCMI instance if it hasn't been initialized already.
        """
        if cls._dcmi_instance is None:
            if lib_path:
                cls._dcmi_instance = DcmiWrapper(lib_path)
                if cls._dcmi_instance is None:
                    logger.error("[DcmiManager] Failed to initialize DCMI.")
                    return False
                logger.info("[DcmiManager] DCMI initialized successfully.")
                return True
            logger.error("[DcmiManager] Invalid library path provided.")
            return False
        logger.info("[DcmiManager] DCMI already initialized.")
        return True

    @classmethod
    def get_dcmi_instance(cls):
        """
        Retrieves the currently initialized DCMI instance.
        """
        return cls._dcmi_instance
