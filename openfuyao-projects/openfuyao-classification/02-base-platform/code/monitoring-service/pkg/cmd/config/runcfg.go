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

// Package config provides the configuration structures and logic necessary
// to set up and manage the runtime environment for the monitoring service.
// This package encapsulates all the necessary details to configure and
// validate the settings required to run the service effectively.
package config

import (
	"openfuyao.com/monitoring-service/pkg/client/prometheus"
	"openfuyao.com/monitoring-service/pkg/server/runtime"
)

// RunConfig holds the configuration settings necessary to start and run the server and monitoring systems.
type RunConfig struct {
	Server            *runtime.Server
	MonitoringOptions *prometheus.Options
}

// NewRunConfig initializes a new RunConfig with default server and Prometheus endpoint.
func NewRunConfig() *RunConfig {
	return &RunConfig{
		Server:            runtime.NewServer(),
		MonitoringOptions: prometheus.NewPrometheusOptions(),
	}
}
