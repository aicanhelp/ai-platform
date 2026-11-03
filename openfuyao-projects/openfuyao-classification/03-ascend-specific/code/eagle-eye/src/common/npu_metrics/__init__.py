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
NPU metrics common library.
"""

from common.npu_metrics.dcmi.dcmi import DcmiManager
from common.npu_metrics.constants import (
    DCMI_LIBRARY_NAME,
    HIAI_MAX_CARD_NUM,
    HIAI_MAX_DEVICE_NUM,
    RET_INVALID_VALUE,
    RET_NO_FUNCTION,
)

__all__ = [
    "DcmiManager",
    "DCMI_LIBRARY_NAME",
    "HIAI_MAX_CARD_NUM",
    "HIAI_MAX_DEVICE_NUM",
    "RET_INVALID_VALUE",
    "RET_NO_FUNCTION",
]
