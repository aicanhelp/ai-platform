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
NPU RoCE metrics definitions.
"""

from network_performance_exporter.metrics.base import MetricDefinition, MetricType

# NPU RoCE physical link status
NPU_ROCE_LINK_STATE = MetricDefinition(
    name="npu_roce_link_state",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card physical link state (1=UP, 0=DOWN)",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE link UP count
NPU_ROCE_LINK_UP_COUNT = MetricDefinition(
    name="npu_roce_link_up_count",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card link UP count (reflects link stability)",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE actual transmission rate
NPU_ROCE_TX_RATE_MBPS = MetricDefinition(
    name="npu_roce_tx_rate_mbps",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card actual transmission rate in MB/s",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE actual reception rate
NPU_ROCE_RX_RATE_MBPS = MetricDefinition(
    name="npu_roce_rx_rate_mbps",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card actual reception rate in MB/s",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE available bandwidth for transmission
NPU_ROCE_TX_AVAILABLE_BANDWIDTH_MBPS = MetricDefinition(
    name="npu_roce_tx_available_bandwidth_mbps",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card available bandwidth for transmission in MB/s",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE available bandwidth for reception
NPU_ROCE_RX_AVAILABLE_BANDWIDTH_MBPS = MetricDefinition(
    name="npu_roce_rx_available_bandwidth_mbps",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card available bandwidth for reception in MB/s",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE packet loss rate
NPU_ROCE_PACKET_LOSS_RATE = MetricDefinition(
    name="npu_roce_packet_loss_rate",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card packet loss rate (0-1)",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# NPU RoCE retransmit rate
NPU_ROCE_RETRANSMIT_RATE = MetricDefinition(
    name="npu_roce_retransmit_rate",
    metric_type=MetricType.GAUGE,
    help_text="NPU RoCE network card retransmit rate (0-1)",
    labels=["node", "logic_id", "phy_id", "card_id", "device_id"]
)

# All NPU RoCE metrics
ALL_NPU_ROCE_METRICS = [
    NPU_ROCE_LINK_STATE,
    NPU_ROCE_LINK_UP_COUNT,
    NPU_ROCE_TX_RATE_MBPS,
    NPU_ROCE_RX_RATE_MBPS,
    NPU_ROCE_TX_AVAILABLE_BANDWIDTH_MBPS,
    NPU_ROCE_RX_AVAILABLE_BANDWIDTH_MBPS,
    NPU_ROCE_PACKET_LOSS_RATE,
    NPU_ROCE_RETRANSMIT_RATE,
]

# Mapping of data source to metrics
METRICS_SOURCE_MAP = {
    "npu_roce": ALL_NPU_ROCE_METRICS,
}
