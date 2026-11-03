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
This module defines RecoveryStrategy and HardwareRecoveryStrategy classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from dataclasses import dataclass, field

from hardware_diagnosis.event.base import Event
from hardware_diagnosis.event.hardware_event import NPUHardwareEvent
# pylint: disable=invalid-name


class RecoveryStrategy(ABC):
    """
    Recovery strategy base class
    All recovery strategies need to implement this type of interface
    """

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """
        Determines if the current recovery strategy can handle the provided event.
        """
        pass

    @abstractmethod
    def execute(self, event: Event) -> Dict[str, Any]:
        """
        Executes the recovery strategy for the given event.
        """
        pass


class HardwareRecoveryStrategy(RecoveryStrategy, ABC):
    """
    Hardware recovery strategy base class
    """

    def can_handle(self, event: Event) -> bool:
        """
        Checks if this strategy can handle a hardware-related event.
        """
        return isinstance(event, NPUHardwareEvent)

    @abstractmethod
    def execute(self, event: NPUHardwareEvent) -> Dict[str, Any]:
        """
        Executes the hardware-specific recovery strategy for the given event.
        """
        pass


@dataclass
class DiagnosisContext:
    """
    Contains the context for the hardware_diagnosis process, including the event, the current step
    in the process, and any results from the previous steps.
    """
    event: Event
    current_step: str
    result: List[Dict[str, Any]] = field(default_factory=list)
    need_further: bool = True


@dataclass
class DiagnosisResult:
    """
    Represents the final result of a hardware_diagnosis process, including the raw event,
    hardware_diagnosis chain, reasoning, severity, and suggested actions.
    """
    raw_event: Event
    diagnosis_chain: List[str]
    reason: str
    severity: int
    suggestion: List[Any]


def register_stage(cls: Type["DiagnosisStage"]) -> Type["DiagnosisStage"]:
    """
    Registers a subclass of DiagnosisStage to the global registry of stages.
    """
    if not hasattr(cls, "stage_name") or not isinstance(cls.stage_name, str):
        raise ValueError(f" {cls.__name__} must have 'stage_name' attribute")

    stage_name = cls.stage_name
    if stage_name in GLOBAL_STAGE_REGISTRY:
        existing_cls = GLOBAL_STAGE_REGISTRY[stage_name]
        raise ValueError(
            f" '{stage_name}' is occupied by  {existing_cls.__name__}，"
        )
    GLOBAL_STAGE_REGISTRY[stage_name] = cls
    return cls


class DiagnosisStage(ABC):
    """
    Base class for hardware_diagnosis stages.
    """
    stage_name: str

    @abstractmethod
    def diagnosis(self, context: DiagnosisContext) -> DiagnosisContext:
        """
        Executes the hardware_diagnosis process for the given context.
        """
        pass


GLOBAL_STAGE_REGISTRY: Dict[str, Type["DiagnosisStage"]] = {}
