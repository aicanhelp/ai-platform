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
This module provides functions for managing and verifying the paths, ownership,
and permissions of shared libraries, particularly for NPU-related libraries.
"""

import subprocess
import os

from typing import Tuple, Union

MAX_PATH_DEPTH = 20
DEFAULT_WRITE_FILE_MODE = 0o022
ROOT_UID = 0


def get_lib_from_ld_cmd(library_name: str) -> Tuple[str, Union[Exception, None]]:
    """
    Retrieves the absolute path of a shared library by checking system configurations.

    This function currently returns a hardcoded path, but it can be extended
    to find the library dynamically using system tools like `ldconfig`.
    """
    # library_abs_name, error = parse_lib_from_ld_cmd(library_name)
    # if error:
    #     return "", error
    #
    # abs_lib_path, error = check_abs_path(library_abs_name)
    # if error:
    #     return "", error

    abs_lib_path = "/usr/local/Ascend/driver/lib64/driver/libdcmi.so"

    return abs_lib_path, None


def check_abs_path(lib_path: str) -> Tuple[str, Union[Exception, None]]:
    """
    Checks the absolute path of the given library for ownership and permissions.
    """
    abs_lib_path, error = check_owner_and_permission(lib_path, DEFAULT_WRITE_FILE_MODE, ROOT_UID)
    if error:
        return "", error

    count = 0
    f_path = abs_lib_path
    while True:
        if count >= MAX_PATH_DEPTH:
            return "", RuntimeError("Maximum path depth reached.")

        count += 1
        if f_path == "/":
            return abs_lib_path, None

        f_path = os.path.dirname(f_path)

        _, dir_error = check_owner_and_permission(f_path, DEFAULT_WRITE_FILE_MODE, ROOT_UID)
        if dir_error:
            return "", dir_error


def check_owner_and_permission(file_path: str, mode: int, uid: int) -> Tuple[str, Union[Exception, None]]:
    """
    Verifies the ownership and permissions of the specified file.
    """
    if not file_path:
        return "", ValueError("empty path")

    abs_path = os.path.abspath(file_path)
    resolved_path = os.path.realpath(abs_path)

    if not os.path.exists(resolved_path):
        return "", FileNotFoundError(f"resolved path does not exist: {resolved_path}")

    if abs_path != resolved_path:
        try:
            file_info = os.lstat(abs_path)
            if file_info.st_uid != uid:
                return "", PermissionError(f"Symlink {abs_path} is not owned by UID {uid}.")
        except OSError as e:
            return "", PermissionError(f"Failed to check symlink {abs_path}: {e}")

    try:
        file_info = os.stat(resolved_path)
    except OSError as e:
        return "", FileNotFoundError(f"Failed to stat {resolved_path}: {e}")

    if file_info.st_uid != uid:
        return "", PermissionError(f"File {resolved_path} is not owned by UID {uid}.")

    if (file_info.st_mode & mode) != 0:
        return "", PermissionError(f"File {resolved_path} does not have the expected permissions.")

    return resolved_path, None


def parse_lib_from_ld_cmd(library_name: str) -> Tuple[str, Union[Exception, None]]:
    """
    Parses the output of the `ldconfig --print-cache` command
    to find the location of a shared library.
    """
    try:
        result = subprocess.run(
            ["/sbin/ldconfig", "--print-cache"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        return "", RuntimeError(f"Error executing ldconfig: {e.stderr.strip()}")

    for line in result.stdout.splitlines():
        ld_info = line.split("=>")
        if len(ld_info) < 2:
            continue

        lib_names = ld_info[0].split()
        for index, lib_name in enumerate(lib_names):
            if index >= MAX_PATH_DEPTH:
                break
            if len(lib_name.strip()) == 0:
                continue

            if lib_name.strip() == library_name.strip():
                return ld_info[1].strip(), None

    return "", RuntimeError(f"Can't find valid lib: {library_name}")
