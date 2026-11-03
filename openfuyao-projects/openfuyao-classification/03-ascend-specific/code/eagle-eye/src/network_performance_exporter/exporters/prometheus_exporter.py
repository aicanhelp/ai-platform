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
Prometheus metrics exporter.
"""

import logging
import os
from typing import Any, Dict

from prometheus_client import Counter, Gauge, CollectorRegistry

from network_performance_exporter.metrics.rdma_metrics import ALL_RDMA_METRICS, METRICS_SOURCE_MAP as RDMA_SOURCE_MAP
from network_performance_exporter.metrics.npu_roce_metrics import ALL_NPU_ROCE_METRICS, METRICS_SOURCE_MAP as NPU_ROCE_SOURCE_MAP
from network_performance_exporter.metrics.disk_metrics import ALL_DISK_METRICS, METRICS_SOURCE_MAP as DISK_SOURCE_MAP
from network_performance_exporter.metrics.base import MetricType

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)

# Combine all source mappings
ALL_METRICS_SOURCE_MAP = {**RDMA_SOURCE_MAP, **NPU_ROCE_SOURCE_MAP, **DISK_SOURCE_MAP}


class PrometheusExporter:
    """Exporter for Prometheus metrics."""

    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self._metrics = {}
        self._metric_labels = {}  # Store label names for each metric
        self._active_labels = {}  # Track active label tuples per metric: {metric_name: set of tuples}
        self._init_metrics()

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        # Initialize RDMA metrics
        for metric_def in ALL_RDMA_METRICS:
            self._create_metric(metric_def)

        # Initialize NPU RoCE metrics
        for metric_def in ALL_NPU_ROCE_METRICS:
            self._create_metric(metric_def)

        # Initialize disk metrics
        for metric_def in ALL_DISK_METRICS:
            self._create_metric(metric_def)

    def _create_metric(self, metric_def):
        """
        Create a Prometheus metric based on definition.

        Args:
            metric_def: MetricDefinition instance
        """
        try:
            if metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_def.name,
                    metric_def.help_text,
                    labelnames=metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_def.name,
                    metric_def.help_text,
                    labelnames=metric_def.labels,
                    registry=self.registry
                )
            else:
                logger.warning("Unsupported metric type: %s", metric_def.metric_type)
                return

            self._metrics[metric_def.name] = metric
            self._metric_labels[metric_def.name] = metric_def.labels
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to create metric %s: %s", metric_def.name, str(e))

    def update(self, metrics_data: Dict[str, Any]):
        """
        Update Prometheus metrics with collected data.

        Args:
            metrics_data: Dictionary with collected metrics keyed by source
        """
        for source, data in metrics_data.items():
            if source not in ALL_METRICS_SOURCE_MAP:
                logger.warning("Unknown metrics source: %s", source)
                continue

            self._update_metrics_for_source(source, data)

    def _update_metrics_for_source(self, source: str, data: Dict[str, Any]):
        """
        Update metrics for a specific source using metric definitions.

        Args:
            source: Data source name (e.g., "rdma", "npu_roce", "disk")
            data: Collected data for this source
        """
        metric_defs = ALL_METRICS_SOURCE_MAP.get(source, [])

        # Snapshot active labels before this round to detect stale entries
        previous_labels = {
            metric_def.name: set(self._active_labels.get(metric_def.name, set()))
            for metric_def in metric_defs
        }

        # Clear active labels for this source so only this round's sets are tracked
        for metric_def in metric_defs:
            self._active_labels[metric_def.name] = set()

        # Handle different data structures based on source
        if source == "rdma":
            # RDMA: {device: {port: {metrics}}}
            for device, ports_data in data.items():
                for port, metrics in ports_data.items():
                    labels = {
                        "node": metrics.get("node", ""),
                        "device": metrics.get("device", device),
                        "port": str(metrics.get("port", port)),
                    }
                    self._update_metric_values(metric_defs, metrics, labels)

        elif source == "npu_roce":
            # NPU RoCE: {logic_id: {metrics}}
            for logic_id, metrics in data.items():
                labels = {
                    "node": metrics.get("node", ""),
                    "logic_id": str(metrics.get("logic_id", logic_id)),
                    "phy_id": str(metrics.get("phy_id", "")),
                    "card_id": str(metrics.get("card_id", "")),
                    "device_id": str(metrics.get("device_id", "")),
                }
                self._update_metric_values(metric_defs, metrics, labels)

        elif source == "disk":
            # Disk: {device: {metrics}}
            for device, metrics in data.items():
                labels = {
                    "node": metrics.get("node", ""),
                    "device": metrics.get("device", device),
                }
                self._update_metric_values(metric_defs, metrics, labels)

        # Remove stale labels for devices/ports that no longer exist
        self._remove_stale_labels(metric_defs, previous_labels)

    def _remove_stale_labels(self, metric_defs, previous_labels: dict):
        """
        Remove label combinations that no longer exist in the current collection.

        Args:
            metric_defs: List of MetricDefinition objects for this source
            previous_labels: Snapshot of active labels from before this round
        """
        for metric_def in metric_defs:
            metric_name = metric_def.name
            current = self._active_labels.get(metric_name, set())
            stale = previous_labels.get(metric_name, set()) - current
            if not stale:
                continue

            metric = self._metrics.get(metric_name)
            if metric is None:
                continue

            for label_values in stale:
                try:
                    metric.remove(*label_values)
                    logger.debug("Removed stale label %s from metric %s", label_values, metric_name)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Failed to remove stale label %s from %s: %s",
                                   label_values, metric_name, str(e))

    def _update_metric_values(self, metric_defs, metrics: Dict[str, Any], labels: Dict[str, str]):
        """
        Update metric values based on definitions.

        Args:
            metric_defs: List of MetricDefinition objects
            metrics: Collected metric values
            labels: Label values for the metrics
        """
        for metric_def in metric_defs:
            if metric_def.name in metrics:
                self._set_metric(metric_def.name, metrics[metric_def.name], labels)

    def _set_metric(self, metric_name: str, value: float, labels: Dict[str, str]):
        """
        Set a metric value.

        Args:
            metric_name: Name of the metric
            value: Value to set
            labels: Dictionary of label values
        """
        try:
            metric = self._metrics.get(metric_name)
            if metric is None:
                logger.warning("Metric %s not found", metric_name)
                return

            label_names = self._metric_labels.get(metric_name, [])
            label_values = [labels.get(label, "") for label in label_names]
            if metric_name not in self._active_labels:
                self._active_labels[metric_name] = set()
            self._active_labels[metric_name].add(tuple(label_values))
            metric.labels(*label_values).set(value)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to set metric %s: %s", metric_name, str(e))

    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus registry."""
        return self.registry
