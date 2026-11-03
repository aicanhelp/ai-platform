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
RDMA metrics definitions.
"""

from network_performance_exporter.metrics.base import MetricDefinition, MetricType

# RDMA link state (1=ACTIVE, 0=DOWN)
RDMA_LINK_STATE = MetricDefinition(
    name="rdma_link_state",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card protocol layer link state (1=ACTIVE, 0=DOWN)",
    labels=["node", "device", "port"]
)

# RDMA physical link state (1=LinkUp, 0=LinkDown)
RDMA_PHYSICAL_LINK_STATE = MetricDefinition(
    name="rdma_physical_link_state",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card physical link state (1=LinkUp, 0=LinkDown)",
    labels=["node", "device", "port"]
)

# RDMA cumulative transmitted data
RDMA_TX_BYTES_TOTAL = MetricDefinition(
    name="rdma_tx_bytes_total",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card cumulative transmitted data in bytes",
    labels=["node", "device", "port"]
)

# RDMA cumulative received data
RDMA_RX_BYTES_TOTAL = MetricDefinition(
    name="rdma_rx_bytes_total",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card cumulative received data in bytes",
    labels=["node", "device", "port"]
)

# RDMA actual transmission rate
RDMA_TX_RATE_MBPS = MetricDefinition(
    name="rdma_tx_rate_mbps",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card actual transmission rate in MB/s",
    labels=["node", "device", "port"]
)

# RDMA actual reception rate
RDMA_RX_RATE_MBPS = MetricDefinition(
    name="rdma_rx_rate_mbps",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card actual reception rate in MB/s",
    labels=["node", "device", "port"]
)

# RDMA available bandwidth for transmission
RDMA_TX_AVAILABLE_BANDWIDTH_MBPS = MetricDefinition(
    name="rdma_tx_available_bandwidth_mbps",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card available bandwidth for transmission in MB/s",
    labels=["node", "device", "port"]
)

# RDMA available bandwidth for reception
RDMA_RX_AVAILABLE_BANDWIDTH_MBPS = MetricDefinition(
    name="rdma_rx_available_bandwidth_mbps",
    metric_type=MetricType.GAUGE,
    help_text="RDMA network card available bandwidth for reception in MB/s",
    labels=["node", "device", "port"]
)

# All RDMA metrics
ALL_RDMA_METRICS = [
    RDMA_LINK_STATE,
    RDMA_PHYSICAL_LINK_STATE,
    RDMA_TX_BYTES_TOTAL,
    RDMA_RX_BYTES_TOTAL,
    RDMA_TX_RATE_MBPS,
    RDMA_RX_RATE_MBPS,
    RDMA_TX_AVAILABLE_BANDWIDTH_MBPS,
    RDMA_RX_AVAILABLE_BANDWIDTH_MBPS,
]

# Mapping of data source to metrics
METRICS_SOURCE_MAP = {
    "rdma": ALL_RDMA_METRICS,
}
