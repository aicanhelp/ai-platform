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
This module defines NPUHardwareRecovery class.
"""

from typing import Dict, Any

from hardware_diagnosis.recovery.base import HardwareRecoveryStrategy
from hardware_diagnosis.event.hardware_event import NPUHardwareEvent
from hardware_diagnosis.constant.constant import ERROR_CODE_MAP


class NPUHardwareRecovery(HardwareRecoveryStrategy):
    """
    This class defines the functions to judge if the event can resolve and how to execute.
    """
    def can_handle(self, event: NPUHardwareEvent) -> bool:
        return event.error_code in ERROR_CODE_MAP

    def execute(self, event: NPUHardwareEvent) -> Dict[str, Any]:
        try:
            # ray.nodes()[node_id]["resources"][f"npu:{event.logic_id}"] = 0.0
            action = ERROR_CODE_MAP[event.error_code]
            return {
                "status": "success",
                "reason": event.details,
                "event": event,
                "action": action,
            }
        except Exception as e:  # pylint: disable=broad-except
            return {
                "status": "failed",
                "event": event,
                "error": str(e),
            }
