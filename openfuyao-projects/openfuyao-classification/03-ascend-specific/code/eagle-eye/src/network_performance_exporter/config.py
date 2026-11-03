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
Configuration management for network performance exporter.
"""

import argparse
import os
from dataclasses import dataclass

# NATS configuration (from nats_communicator)
NATS_TOPIC_PREFIX = "metrics.network"


@dataclass
class Config:
    """Configuration for network performance exporter."""

    # Prometheus configuration
    prometheus_port: int
    prometheus_host: str

    # Collection configuration
    collect_interval: int

    # Node configuration
    node_name: str


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Network Performance Exporter"
    )

    parser.add_argument(
        '--web-listen-port',
        type=int,
        default=8000,
        help='HTTP server listen port (default: %(default)s)'
    )

    parser.add_argument(
        '--collector-interval',
        type=int,
        default=15,
        help='Metrics collection interval in seconds (default: %(default)s)'
    )

    return parser


def get_config() -> Config:
    """Parse command-line arguments and return Config."""
    parser = create_argument_parser()
    args = parser.parse_args()

    return Config(
        prometheus_port=args.web_listen_port,
        prometheus_host="0.0.0.0",
        collect_interval=args.collector_interval,
        node_name=os.getenv("NODE_NAME", "")
    )



