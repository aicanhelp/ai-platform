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
This module defines the base class for NATS client communication,
including methods for connecting and disconnecting from the NATS server.
"""

import logging
import os

from nats.aio.client import Client as NATS_Client
from nats_communicator.constants import NATS_SERVER

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class NatsClientBase:
    """
    A base class for managing NATS client connections.
    """
    def __init__(self):
        self.client = NATS_Client()

    async def connect(self):
        """
        Asynchronously connects to the NATS server using the server URL defined in `NATS_SERVER`.
        """
        try:
            await self.client.connect(
                NATS_SERVER,
                max_reconnect_attempts=3,
                reconnect_time_wait=5
            )
            logger.info("Connected to NATS server at %s", NATS_SERVER)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to connect to NATS server: %s", str(e))

    async def disconnect(self):
        """
        Asynchronously disconnects from the NATS server.
        """

        if not self.client.is_connected:
            logger.info("NATS client not connected, skipping disconnect")
            return

        try:
            await self.client.drain()
            logger.info("Disconnected from NATS server")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to disconnect from NATS: %s", str(e))
