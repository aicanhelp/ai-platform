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
This module defines the `NatsSubscriberWithProcessing` class that subscribes to NATS messages, 
processes events, and triggers recovery strategies based on detected issues.
"""

import json
import asyncio

from logger.setup_logger import setup_logger
from hardware_diagnosis.detector.npu.detector import NpuDetector
from hardware_diagnosis.recovery.factory import RecoveryStrategyFactory
from nats_communicator.subscriber import NatsSubscriber
from utils.utils import detect_devices

logger = setup_logger()


class NatsSubscriberWithProcessing(NatsSubscriber):
    """
    This class extends `NatsSubscriber` to subscribe to NATS messages,
    process the messages through an NPU detector,
    and apply appropriate recovery strategies using a factory pattern.
    """

    def __init__(self):
        super().__init__()
        self.npu_detector = NpuDetector()
        self.npu_detector.set_event_callback(self.npu_detector.process_events)
        self.recovery_factory = RecoveryStrategyFactory()
        self.is_subscribed = False

    async def message_handler(self, msg):
        """
        This function handles incoming messages from NATS, decodes the message data,
        parses it into a dictionary,
        and starts the event detection process through the NPU detector.
        """
        data = msg.data.decode()
        data_dict = self.parse_message(data)

        self.npu_detector.start(data_dict)
        self.get_recovery(self.npu_detector.filter_events)

    def parse_message(self, data):
        """
        Parses the incoming message data from JSON format to a Python dictionary.
        """
        try:
            return json.loads(data)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to parse message: %s", e)
            return {}

    async def start(self):
        """
        Connects to NATS and subscribes to the specified topic if not already subscribed.
        """
        if not self.is_subscribed:
            await self.connect()
            await self.subscribe(self.message_handler)

            self.is_subscribed = True

    def get_recovery(self, events):
        """
        This method iterates over the detected events, retrieves the appropriate recovery strategy from the factory,
        and executes the strategy. The results of the recovery actions are logged.
        """
        for event in events:
            try:
                strategy = self.recovery_factory.get_strategy(event)
                if strategy:
                    info = strategy.execute(event)
                    logger.info("Event %s processed: %s", event.event_id, info)

            except KeyError as ke:
                logger.error("Event %s has invalid structure: %s", event.event_id, ke)
            except Exception as ex:  # pylint: disable=broad-except
                logger.error("Failed to process event %s, error: %s", event.event_id, ex, exc_info=True)


if __name__ == "__main__":
    async def main():
        """
        Initializes and starts the NATS subscriber with processing. The subscriber listens to NATS messages and processes
        them using the NPU detector and recovery strategies.
        """
        devices = detect_devices()
        if devices["npu"] or devices["gpu"]:
            if devices["npu"]:
                logger.info("NPU detected.")
                nats_subscriber_with_processing = NatsSubscriberWithProcessing()
                await nats_subscriber_with_processing.start()

            if devices["gpu"]:
                logger.info("GPU detected.")
        else:
            logger.info("No NPU or GPU found.")

        while True:
            await asyncio.sleep(1)


    asyncio.run(main())
