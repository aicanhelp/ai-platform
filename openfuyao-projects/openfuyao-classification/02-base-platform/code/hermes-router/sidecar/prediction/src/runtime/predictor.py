# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Predictor protocol and runtime errors."""

from __future__ import annotations

from typing import Protocol

from domain.models import PredictInput, PredictOutput


class FailedPreconditionError(RuntimeError):
    """Raised when a predictor is configured but not ready to serve."""


class Predictor(Protocol):
    model_version: str

    def predict(self, target_model: str, inputs: list[PredictInput]) -> list[PredictOutput]:
        """Return prediction outputs for valid rows."""
