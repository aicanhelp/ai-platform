# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module defines RecoveryStrategyFactory class.
"""

from typing import List, Optional

from hardware_diagnosis.event.base import Event
from hardware_diagnosis.recovery.base import RecoveryStrategy
from hardware_diagnosis.recovery.hardware_strategies.npu.npu_recovery import NPUHardwareRecovery


class RecoveryStrategyFactory:
    """
    This class is the strategy pattern.
    """
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = [
            NPUHardwareRecovery(),
        ]

    def register_strategy(self, strategy: RecoveryStrategy):
        """
        Registers a new recovery strategy to the factory.
        """
        self.strategies.append(strategy)

    def get_strategy(self, event: Event) -> Optional[RecoveryStrategy]:
        """
        Retrieves the appropriate recovery strategy for a given event.
        """
        for strategy in self.strategies:
            if strategy.can_handle(event):
                return strategy
        return None
