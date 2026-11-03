/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config"
	logsapi "k8s.io/component-base/logs/api/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ColocationConfiguration 包含所有必须的colocation-service服务器的配置。
type ColocationConfiguration struct {
	metav1.TypeMeta

	// bindAddress 服务器的监听地址
	BindAddress string
	// healthzBindAddress 健康检查服务器的监听地址
	HealthzBindAddress string
	// metricsBindAddress 指标服务器的监听
	MetricsBindAddress string
	// BindAddressHardFail, 如果true,当colocation-service绑定端口失败将退出
	BindAddressHardFail bool
	// enableProfiling 通过/debug/pprof接口使能性能分析
	EnableProfiling bool
	// clientConnection 指定kubeconfig文件
	ClientConnection componentbaseconfig.ClientConnectionConfiguration
	// oomScoreAdj 是colocation-service进程的oom-score-adj值。
	OOMScoreAdj *int32
	// configSyncPeriod 从apiserver刷新配置的周期
	ConfigSyncPeriod metav1.Duration
	// ShowHiddenMetricsForVersion 是你想要显示隐藏指标的版本。
	ShowHiddenMetricsForVersion string
	// Logging 指定日志的Options
	Logging logsapi.LoggingConfiguration
}
