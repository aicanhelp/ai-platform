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
Logging configuration for eagle-eye modules.
"""

import logging
import os

from logging.handlers import RotatingFileHandler


def setup_logger():
    """
    Sets up logging configuration, including file and console handlers.
    The log file is dynamically named based on the module name.
    """
    module_name = os.getenv("MODULE_NAME", "default")

    logger = logging.getLogger(module_name)

    if not logger.hasHandlers():
        log_dir = f"/var/log/{module_name}"
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = f"{log_dir}/{module_name}.log"
        log_handler = RotatingFileHandler(log_file_path, maxBytes=40 * 1024 * 1024, backupCount=0)
        log_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(log_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)

    return logger
