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
This module defines the NatsSubscriber class, which is responsible for subscribing to
a NATS topic and processing messages received on that topic.
"""

import logging
import os

from nats_communicator.base import NatsClientBase
from nats_communicator.constants import TOPIC

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class NatsSubscriber(NatsClientBase):
    """
    NatsSubscriber is a class that handles subscribing to a NATS topic and processing
    messages published to that topic.
    """

    async def subscribe(self, message_handler, topic: str = TOPIC):
        """
        A class for subscribing to a specific NATS topic and processing incoming messages.
        """
        try:
            async def on_message(msg):
                logger.info("Received message on topic %s", topic)
                await message_handler(msg)

            if not self.client.is_connected:
                await self.connect()

            await self.client.subscribe(topic, cb=on_message)
            logger.info("Successfully subscribed to NATS topic %s", topic)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to subscribe to NATS topic %s: %s", topic, str(e))
