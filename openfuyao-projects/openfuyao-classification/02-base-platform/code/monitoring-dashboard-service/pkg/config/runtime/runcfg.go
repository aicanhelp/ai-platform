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

// Package runtime contains runtime configuration for the server.
package runtime

import (
	"monitoring-dashboard-service/pkg/config/k8s"
)

// RunConfig encapsulates the runtime configuration of the server,
// potentially including other settings beyond the basic server configuration.
type RunConfig struct {
	Server        *ServerConfig
	KubernetesCfg *k8s.KubernetesCfg
}

// NewRunConfig creates a RunConfig struct with a default Server configuration.
func NewRunConfig() *RunConfig {
	return &RunConfig{
		Server:        NewServerConfig(),
		KubernetesCfg: k8s.NewKubernetesCfg(),
	}
}
