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
This module defines the NatsPublisher class,
which is responsible for publishing messages to a NATS topic.
"""

import json
import logging
import os

from nats_communicator.base import NatsClientBase
from nats_communicator.constants import TOPIC

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class NatsPublisher(NatsClientBase):
    """
    A class responsible for publishing messages to a NATS topic.
    """
    async def publish(self, message: dict, topic: str = TOPIC):
        """
        Publishes a given message to a NATS topic after serializing it to JSON.
        """
        if self.client.is_connected:
            try:
                message_json = json.dumps(message)
                message_json_encode = message_json.encode()
                await self.client.publish(topic, message_json_encode)
                logger.info("Successfully published results to NATS topic %s: %s", topic, message_json_encode)
            except Exception as e:    # pylint: disable=broad-except
                logger.error("Failed to publish message to NATS: %s", str(e))
        else:
            logger.info("NATS client not connected, skipping publish")
