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
Common utilities for file path checking and size-limited writing.
"""

import os

from io import BytesIO


def check_path(path: str):
    """
    Checks and resolves the given file path, ensuring it exists and is not a symlink.
    """
    if not path:
        return path, ValueError("The provided path is empty.")

    origin = path
    while not os.path.lexists(path):
        path = os.path.dirname(path)
        if path == "." or not path:
            return "", FileNotFoundError("Path not found: %s", origin)

    try:
        abs_path = os.path.abspath(path)
        resolved_path = os.path.realpath(abs_path)
        if abs_path != resolved_path:
            return "", RuntimeError("Can't support symlinks")
    except Exception as e:  # pylint: disable=broad-except
        error_msg = str(e)
        return "", RuntimeError("Failed to get absolute path or real path: %s", error_msg)

    try:
        abs_origin = os.path.abspath(origin)
    except Exception as e:  # pylint: disable=broad-except
        return "", RuntimeError(f"get the absolute path failed: {e}")

    return abs_origin, None


class LimitedWriter:
    """
    LimitedWriter is an in-memory writer with a fixed byte limit.
    """

    def __init__(self, limit: int):
        self._buf = BytesIO()
        self._limit = limit
        self._written = 0

    def write(self, data: bytes) -> int:
        """
            Writes data to the buffer, respecting the buffer's size limit.
        """
        if self._written >= self._limit:
            return 0
        room = self._limit - self._written
        chunk = data[:room]
        n = self._buf.write(chunk)
        self._written += n
        return n

    def get_buffer_bytes(self) -> bytes:
        """
        Retrieves the current contents of the buffer as bytes.
        """
        return self._buf.getvalue()
