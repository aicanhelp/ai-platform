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
This module defines the NPUDeviceInfoCollector class,
which is responsible for collecting information about the NPU devices,
including details about the cards and devices in the system. It interacts with 
the NPU utility functions to gather the device information and stores the results in a cache.
"""

import logging
import os

from common.base.base_info_collector import BaseInfoCollector
from common.npu_metrics.utils import (get_card_list, get_device_num_in_card,
                                      get_device_logic_id, get_phy_id)

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)

# Cache keys used by this collector (clear only these to avoid wiping container info)
_DEVICE_INFO_KEYS = ("logic_id", "card_id", "device_id", "phy_id")


class NPUDeviceInfoCollector(BaseInfoCollector):
    """
    Collects information about NPU devices, including card IDs, device IDs, and logic IDs.
    """

    def collect_info(self):
        for key in _DEVICE_INFO_KEYS:
            self.cache.clear(key)

        _, card_list, _, error = get_card_list()
        if error:
            logger.error("[NPUDeviceInfoCollector] %s", error)
            return

        for card_id in card_list:
            device_count, error = get_device_num_in_card(card_id)
            if error:
                logger.error("[NPUDeviceInfoCollector] %s", error)
                continue

            for device_id in range(device_count):
                logic_id, error = get_device_logic_id(card_id, device_id)
                if error:
                    logger.error("[NPUDeviceInfoCollector] %s", error)
                    continue

                phy_id, error = get_phy_id(logic_id)
                if error:
                    logger.error("[NPUDeviceInfoCollector] %s", error)
                    continue

                device_info = {
                    "card_id": card_id,
                    "device_id": device_id,
                    "phy_id": phy_id,
                    "logic_id": logic_id
                }

                self.update_cache(device_info)
