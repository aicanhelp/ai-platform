# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Compatibility helpers for generated tokenizer protobuf modules."""

from __future__ import annotations

import sys

from . import tokenizer_pb2 as _tokenizer_pb2

# grpc_tools generates absolute sibling imports for Python modules. Register the
# package-local protobuf module under the top-level name so installed package
# layouts continue to work without patching generated files.
sys.modules.setdefault("tokenizer_pb2", _tokenizer_pb2)
