/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package config provides configuration management for the server.
package config

import (
	"openfuyao-rayservice/pkg/client/k8s"
	db_runtime "openfuyao-rayservice/pkg/database/runtime"
	"openfuyao-rayservice/pkg/server/runtime"
)

// RunConfig holds config for the server
type RunConfig struct {
	Server        *runtime.ServerConfig
	DbConfig      *db_runtime.DbConfig
	KubernetesCfg *k8s.KubernetesCfg
}

// NewRunConfig creates a new RunConfig with default values
func NewRunConfig() *RunConfig {
	return &RunConfig{
		Server:        runtime.NewServerConfig(),
		DbConfig:      db_runtime.NewDbConfig(),
		KubernetesCfg: k8s.NewKubernetesCfg(),
	}
}

// Validate the RunConfig
func (cfg *RunConfig) Validate() []error {
	var errs []error
	errs = append(errs, cfg.Server.Validate()...)
	errs = append(errs, cfg.KubernetesCfg.Validate()...)
	return errs
}
