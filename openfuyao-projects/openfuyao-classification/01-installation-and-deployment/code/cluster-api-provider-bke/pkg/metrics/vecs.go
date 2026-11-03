/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

const (
	PhaseDurationSeconds            = "phase_duration_seconds"
	ClusterUnhealthyCount           = "cluster_unhealthy_count"
	ClusterReadyStatus              = "cluster_ready_status"
	ClusterBootstrapDurationSeconds = "cluster_bootstrap_duration_seconds"
	NodeBootstrapDurationSeconds    = "node_bootstrap_duration_seconds"
	NodeBootstrapFailedCount        = "node_bootstrap_failed_count"
	NodeBootstrapSuccessCount       = "node_bootstrap_success_count"

	AddonInstallDurationSeconds = "addon_install_duration_seconds"
	AddonInstallFailedCount     = "addon_install_failed_count"
	AddonInstallSuccessCount    = "addon_install_success_count"
	AddonInstallCount           = "addon_install_count"
)

// MetricsVectors 提供创建各种度量向量的方法
type MetricsVectors struct{}

// NewMetricsVectors 创建一个新的 MetricsVectors 实例
func NewMetricsVectors() *MetricsVectors {
	return &MetricsVectors{}
}

// PhaseDurationVec 创建阶段持续时间向量
func (mv *MetricsVectors) PhaseDurationVec(clusterKey string) (string, prometheus.Collector) {
	return PhaseDurationSeconds, prometheus.NewGaugeVec(
		newGaugeOpts(PhaseDurationSeconds, clusterKey),
		[]string{"phase", "start_time", "end_time", "describe"},
	)
}

// ClusterUnhealthyCountVec 创建集群不健康计数向量
func (mv *MetricsVectors) ClusterUnhealthyCountVec(clusterKey string) (string, prometheus.Collector) {
	return ClusterUnhealthyCount, prometheus.NewCounterVec(
		newCounterOpts(ClusterUnhealthyCount, clusterKey),
		nil,
	)
}

// ClusterReadyStatusVec 创建集群就绪状态向量
func (mv *MetricsVectors) ClusterReadyStatusVec(clusterKey string) (string, prometheus.Collector) {
	return ClusterReadyStatus, prometheus.NewGaugeVec(
		newGaugeOpts(ClusterReadyStatus, clusterKey),
		nil,
	)
}

// ClusterBootstrapDurationVec 创建集群引导持续时间向量
func (mv *MetricsVectors) ClusterBootstrapDurationVec(clusterKey string) (string, prometheus.Collector) {
	return ClusterBootstrapDurationSeconds, prometheus.NewGaugeVec(
		newGaugeOpts(ClusterBootstrapDurationSeconds, clusterKey),
		[]string{"start_time", "end_time"},
	)
}

// NodeBootstrapDurationVec 创建节点引导持续时间向量
func (mv *MetricsVectors) NodeBootstrapDurationVec(clusterKey string) (string, prometheus.Collector) {
	return NodeBootstrapDurationSeconds, prometheus.NewGaugeVec(
		newGaugeOpts(NodeBootstrapDurationSeconds, clusterKey),
		[]string{"node", "role", "boot_success", "start_time", "end_time"},
	)
}

// NodeBootstrapFailedCountVec 创建节点引导失败计数向量
func (mv *MetricsVectors) NodeBootstrapFailedCountVec(clusterKey string) (string, prometheus.Collector) {
	return NodeBootstrapFailedCount, prometheus.NewCounterVec(
		newCounterOpts(NodeBootstrapFailedCount, clusterKey),
		nil,
	)
}

// NodeBootstrapSuccessCountVec 创建节点引导成功计数向量
func (mv *MetricsVectors) NodeBootstrapSuccessCountVec(clusterKey string) (string, prometheus.Collector) {
	return NodeBootstrapSuccessCount, prometheus.NewCounterVec(
		newCounterOpts(NodeBootstrapSuccessCount, clusterKey),
		nil,
	)
}

// AddonInstallDurationVec 创建插件安装持续时间向量
func (mv *MetricsVectors) AddonInstallDurationVec(clusterKey string) (string, prometheus.Collector) {
	return AddonInstallDurationSeconds, prometheus.NewGaugeVec(
		newGaugeOpts(AddonInstallDurationSeconds, clusterKey),
		[]string{"addon", "start_time", "end_time"},
	)
}

// AddonInstallFailedCountVec 创建插件安装失败计数向量
func (mv *MetricsVectors) AddonInstallFailedCountVec(clusterKey string) (string, prometheus.Collector) {
	return AddonInstallFailedCount, prometheus.NewCounterVec(
		newCounterOpts(AddonInstallFailedCount, clusterKey),
		nil,
	)
}

// AddonInstallSuccessCountVec 创建插件安装成功计数向量
func (mv *MetricsVectors) AddonInstallSuccessCountVec(clusterKey string) (string, prometheus.Collector) {
	return AddonInstallSuccessCount, prometheus.NewCounterVec(
		newCounterOpts(AddonInstallSuccessCount, clusterKey),
		nil,
	)
}

// AddonInstallCountVec 创建插件安装计数向量
func (mv *MetricsVectors) AddonInstallCountVec(clusterKey string) (string, prometheus.Collector) {
	return AddonInstallCount, prometheus.NewCounterVec(
		newCounterOpts(AddonInstallCount, clusterKey),
		nil,
	)
}

// 全局实例，保持向后兼容性
var defaultVectors = NewMetricsVectors()

// 以下函数保持向后兼容性，内部使用默认实例
func PhaseDurationVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.PhaseDurationVec(clusterKey)
}

func ClusterUnhealthyCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.ClusterUnhealthyCountVec(clusterKey)
}

func ClusterReadyStatusVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.ClusterReadyStatusVec(clusterKey)
}

func ClusterBootstrapDurationVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.ClusterBootstrapDurationVec(clusterKey)
}

func NodeBootstrapDurationVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.NodeBootstrapDurationVec(clusterKey)
}

func NodeBootstrapFailedCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.NodeBootstrapFailedCountVec(clusterKey)
}

func NodeBootstrapSuccessCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.NodeBootstrapSuccessCountVec(clusterKey)
}

func AddonInstallDurationVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.AddonInstallDurationVec(clusterKey)
}

func AddonInstallFailedCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.AddonInstallFailedCountVec(clusterKey)
}

func AddonInstallSuccessCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.AddonInstallSuccessCountVec(clusterKey)
}

func AddonInstallCountVec(clusterKey string) (string, prometheus.Collector) {
	return defaultVectors.AddonInstallCountVec(clusterKey)
}

func RegisterBkeVec(collector Collector) {
	collector.RegisterVec(
		PhaseDurationVec,
		ClusterUnhealthyCountVec,
		ClusterReadyStatusVec,
		ClusterBootstrapDurationVec,
		NodeBootstrapDurationVec,
		NodeBootstrapFailedCountVec,
		NodeBootstrapSuccessCountVec,
		AddonInstallDurationVec,
		AddonInstallFailedCountVec,
		AddonInstallSuccessCountVec,
		AddonInstallCountVec,
	)
}
