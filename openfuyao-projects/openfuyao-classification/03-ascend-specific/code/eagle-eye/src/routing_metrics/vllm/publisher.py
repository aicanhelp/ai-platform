# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import logging
import asyncio
import time

from typing import Optional
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from nats_communicator.publisher import NatsPublisher

module_name = os.getenv("MODULE_NAME", "default")

logger = logging.getLogger(module_name)


class CustomStatLogger(StatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.nats_publisher = NatsPublisher()

        asyncio.create_task(self.nats_publisher.connect())

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        metrics = {
            "timestamp": time.time(),
            "node_name": os.getenv("NODE_NAME", ""),
            "pod_ip": os.getenv("POD_IP", ""),
            "model": self.vllm_config.model_config.served_model_name,
            "engine_index": str(self.engine_index),
            "num_requests_waiting": scheduler_stats.num_waiting_reqs,
            "cache_usage_perc": scheduler_stats.kv_cache_usage
        }

        asyncio.create_task(self.nats_publisher.publish(metrics, "eagle_eye.routing_metrics"))

    def log_engine_initialized(self):
        pass
