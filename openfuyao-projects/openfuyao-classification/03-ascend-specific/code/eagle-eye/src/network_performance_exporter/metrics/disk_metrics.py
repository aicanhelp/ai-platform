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
Disk metrics definitions.
"""

from network_performance_exporter.metrics.base import MetricDefinition, MetricType

# Disk read operations completed
DISK_READS_COMPLETED_TOTAL = MetricDefinition(
    name="disk_reads_completed_total",
    metric_type=MetricType.GAUGE,
    help_text="Total number of disk read operations completed",
    labels=["node", "device"]
)

# Disk read bytes
DISK_READ_BYTES_TOTAL = MetricDefinition(
    name="disk_read_bytes_total",
    metric_type=MetricType.GAUGE,
    help_text="Total number of bytes read from disk",
    labels=["node", "device"]
)

# Disk read time
DISK_READ_TIME_MS_TOTAL = MetricDefinition(
    name="disk_read_time_ms_total",
    metric_type=MetricType.GAUGE,
    help_text="Total time spent reading from disk in milliseconds",
    labels=["node", "device"]
)

# Disk write operations completed
DISK_WRITES_COMPLETED_TOTAL = MetricDefinition(
    name="disk_writes_completed_total",
    metric_type=MetricType.GAUGE,
    help_text="Total number of disk write operations completed",
    labels=["node", "device"]
)

# Disk write bytes
DISK_WRITE_BYTES_TOTAL = MetricDefinition(
    name="disk_write_bytes_total",
    metric_type=MetricType.GAUGE,
    help_text="Total number of bytes written to disk",
    labels=["node", "device"]
)

# Disk write time
DISK_WRITE_TIME_MS_TOTAL = MetricDefinition(
    name="disk_write_time_ms_total",
    metric_type=MetricType.GAUGE,
    help_text="Total time spent writing to disk in milliseconds",
    labels=["node", "device"]
)

# Disk read IOPS
DISK_READ_IOPS = MetricDefinition(
    name="disk_read_iops",
    metric_type=MetricType.GAUGE,
    help_text="Disk read operations per second",
    labels=["node", "device"]
)

# Disk write IOPS
DISK_WRITE_IOPS = MetricDefinition(
    name="disk_write_iops",
    metric_type=MetricType.GAUGE,
    help_text="Disk write operations per second",
    labels=["node", "device"]
)

# Disk read throughput
DISK_READ_THROUGHPUT_MB = MetricDefinition(
    name="disk_read_throughput_mb",
    metric_type=MetricType.GAUGE,
    help_text="Disk read throughput in MB/s",
    labels=["node", "device"]
)

# Disk write throughput
DISK_WRITE_THROUGHPUT_MB = MetricDefinition(
    name="disk_write_throughput_mb",
    metric_type=MetricType.GAUGE,
    help_text="Disk write throughput in MB/s",
    labels=["node", "device"]
)

# Disk read utilization
DISK_READ_UTILIZATION = MetricDefinition(
    name="disk_read_utilization",
    metric_type=MetricType.GAUGE,
    help_text="Disk read utilization percentage (0-100%)",
    labels=["node", "device"]
)

# Disk write utilization
DISK_WRITE_UTILIZATION = MetricDefinition(
    name="disk_write_utilization",
    metric_type=MetricType.GAUGE,
    help_text="Disk write utilization percentage (0-100%)",
    labels=["node", "device"]
)

# All disk metrics
ALL_DISK_METRICS = [
    DISK_READS_COMPLETED_TOTAL,
    DISK_READ_BYTES_TOTAL,
    DISK_READ_TIME_MS_TOTAL,
    DISK_WRITES_COMPLETED_TOTAL,
    DISK_WRITE_BYTES_TOTAL,
    DISK_WRITE_TIME_MS_TOTAL,
    DISK_READ_IOPS,
    DISK_WRITE_IOPS,
    DISK_READ_THROUGHPUT_MB,
    DISK_WRITE_THROUGHPUT_MB,
    DISK_READ_UTILIZATION,
    DISK_WRITE_UTILIZATION,
]

# Mapping of data source to metrics
METRICS_SOURCE_MAP = {
    "disk": ALL_DISK_METRICS,
}
