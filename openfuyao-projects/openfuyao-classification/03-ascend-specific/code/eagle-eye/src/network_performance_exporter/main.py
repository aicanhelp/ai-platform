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
Main entry point for network performance exporter.
"""

import asyncio
import os
import signal
import sys
from typing import Optional

from prometheus_client import start_http_server

from logger.setup_logger import setup_logger
from network_performance_exporter.config import get_config
from network_performance_exporter.collector import NetworkPerformanceCollector
from network_performance_exporter.exporters.prometheus_exporter import PrometheusExporter
from network_performance_exporter.exporters.nats_exporter import NatsExporter

module_name = os.getenv("MODULE_NAME", "network_performance_exporter")
logger = setup_logger()


class NetworkPerformanceExporter:
    """Main exporter application."""

    def __init__(self):
        self.config = get_config()
        self.prometheus_exporter: Optional[PrometheusExporter] = None
        self.nats_exporter: Optional[NatsExporter] = None
        self.collector: Optional[NetworkPerformanceCollector] = None

    async def initialize(self):
        """Initialize the exporter."""
        logger.info("Initializing network performance exporter...")
        logger.info("Configuration: listen_address=%s:%d, collect_interval=%ds, node_name=%s",
                    self.config.prometheus_host,
                    self.config.prometheus_port,
                    self.config.collect_interval,
                    self.config.node_name)

        # Initialize Prometheus exporter
        self.prometheus_exporter = PrometheusExporter()
        logger.info("Prometheus exporter initialized")

        # Initialize NATS exporter
        self.nats_exporter = NatsExporter()
        logger.info("NATS exporter initialized")
        await self.nats_exporter.connect()

        # Initialize collector (coordinator)
        self.collector = NetworkPerformanceCollector(
            self.config,
            self.prometheus_exporter,
            self.nats_exporter,
            update_interval=self.config.collect_interval
        )
        logger.info("Network performance collector initialized")

    def start(self):
        """Start the exporter."""
        logger.info("Starting network performance exporter...")

        # Start Prometheus HTTP server
        start_http_server(
            self.config.prometheus_port,
            addr=self.config.prometheus_host,
            registry=self.prometheus_exporter.get_registry()
        )

        logger.info("Prometheus HTTP server started on %s:%d",
                    self.config.prometheus_host, self.config.prometheus_port)

        self.collector.bootstrap()
        logger.info("Collector bootstrap completed")

        self.collector.start()
        logger.info("All collectors started with interval %d seconds", self.config.collect_interval)

        logger.info("Network performance exporter started successfully")

    async def stop(self):
        """Stop the exporter."""
        logger.info("Stopping network performance exporter...")

        # Stop all sub-collectors
        self.collector.stop()
        logger.info("All collectors stopped")

        # Disconnect NATS exporter
        await self.nats_exporter.disconnect()
        logger.info("NATS exporter disconnected")

        logger.info("Network performance exporter stopped")

    async def run(self):
        """Run the exporter."""
        try:
            await self.initialize()
            self.start()  # Synchronous call, no await needed

            # Create stop event for graceful shutdown
            stop_event = asyncio.Event()

            # Register signal handlers in the event loop
            loop = asyncio.get_running_loop()

            def signal_handler():
                logger.info("Received signal, stopping...")
                stop_event.set()

            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

            # Run collection loop until stop signal
            await self.collector.run_forever(stop_event)

        except Exception as e:
            logger.error("Fatal error in network performance exporter: %s", str(e))
            raise
        finally:
            await self.stop()


def main():
    """Main entry point."""
    logger.info("Network performance exporter starting...")

    exporter = NetworkPerformanceExporter()

    # Run the exporter
    try:
        asyncio.run(exporter.run())
    except Exception as e:
        logger.error("Network performance exporter failed to start: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
