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
NATS metrics exporter.
"""

import json
import logging
import os
from typing import Any, Dict

from nats_communicator.base import NatsClientBase
from network_performance_exporter.config import NATS_TOPIC_PREFIX

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)


class NatsExporter(NatsClientBase):
    """Exporter for publishing metrics to NATS."""

    def __init__(self):
        super().__init__()
        self.topic_prefix = NATS_TOPIC_PREFIX

    async def publish(self, metrics_data: Dict[str, Any]):
        """
        Publish all metrics to a single NATS topic.

        Args:
            metrics_data: Dictionary with collected metrics
        """
        if not self.is_connected():
            logger.warning("NATS publisher not connected, skipping publish")
            return

        try:
            message_json = json.dumps(metrics_data)
            message_json_encode = message_json.encode()
            await self.client.publish(self.topic_prefix, message_json_encode)
            logger.info("Successfully published metrics to NATS topic %s", self.topic_prefix)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to publish metrics to NATS: %s", str(e))

    def is_connected(self) -> bool:
        """Check if connected to NATS server."""
        return self.client.is_connected
