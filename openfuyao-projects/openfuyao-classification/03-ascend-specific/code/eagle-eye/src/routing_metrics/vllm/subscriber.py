# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time
import json
import asyncio

from typing import Dict,Any
from nats.aio.client import Client as NATS_Client

from logger.setup_logger import setup_logger

logger = setup_logger()

class RouterSubscriber:
    def __init__(self, server_address: str, topic: str):
        self.server_address = server_address
        self.topic = topic
        self.client = NATS_Client()

        self.message_count = 0
        self.total_latency = 0
        self.last_calculation_time = time.time()

    async def connect(self):
        try:
            await self.client.connect(self.server_address)
            logger.info("Connected to NATS server at %s", self.server_address)
        except Exception as e:
            logger.error("Failed to connect to NATS server: %s", e)
            raise e

    async def subscribe(self, message_handler):
        try:
            async def on_message(msg):
                logger.info("Received message on topic %s", self.topic)
                await message_handler(msg)

            if not self.client.is_connected:
                await self.connect()

            await self.client.subscribe(self.topic, cb=on_message)
            logger.info("Successfully subscribed to NATS topic %s", self.topic)

        except Exception as e:
            logger.error("Failed to subscribe to NATS topic %s: %s", self.topic, e)
            raise e

    async def message_handler(self, msg) -> None:
        current_time = time.time()

        try:
            data = msg.data.decode("utf-8")
        except Exception as exc:    # pylint: disable=broad-except
            logger.error("Failed to decode NATS message: %s", exc)
            return

        data_dict = self.parse_message(data)

        if current_time - self.last_calculation_time <= 1:
            message_timestamp = data_dict.get("timestamp")
            if message_timestamp:
                latency = current_time - message_timestamp
                latency = max(0, latency)
                self.total_latency += latency
                self.message_count += 1
        else:
            throughput = self.message_count
            average_latency = self.total_latency / self.message_count \
                if self.message_count > 0 else 0

            logger.info("Throughput in the last second: %d messages", throughput)
            logger.info("Average latency: %.4f seconds", average_latency)

            self.message_count = 0
            self.total_latency = 0
            self.last_calculation_time = current_time

    def parse_message(self, data: str) -> Dict[str, Any]:
        try:
            return json.loads(data)
        except Exception as exc:    # pylint: disable=broad-except
            logger.error("Failed to parse message: %s, raw: %r", exc, data)
            return {}


if __name__ == "__main__":
    async def main():

        router_subscriber = RouterSubscriber("nats://nats.nats.svc.cluster.local:4222",
                                             "eagle_eye.routing_metrics")
        await router_subscriber.connect()
        await router_subscriber.subscribe(router_subscriber.message_handler)


        while True:
            await asyncio.sleep(1)


    asyncio.run(main())
