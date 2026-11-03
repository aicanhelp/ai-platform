# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Compatibility helpers for generated prediction protobuf modules."""

from __future__ import annotations

import sys

try:
    from . import prediction_pb2 as _prediction_pb2
except ImportError:  # pragma: no cover - generated during codegen
    _prediction_pb2 = None

if _prediction_pb2 is not None:
    sys.modules.setdefault("prediction_pb2", _prediction_pb2)
