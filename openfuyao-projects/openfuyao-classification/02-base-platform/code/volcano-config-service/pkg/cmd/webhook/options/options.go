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

// Package options describe the config for main.go
package options

// Opts path and basic information
type Opts struct {
	DryRun         bool   // dry run
	Healthz        bool   // health check
	Port           int    // webhook nriServer port
	CertFile       string // path to the x509 certificate for https
	KeyFile        string // path to the x509 private key matching `CertFile`
	SidecarCfgFile string // path to sidecar injector configuration file
	SvcNamespace   string // svc namespace
	SvcName        string // svc name
}

// NewOptions new Opts struct
func NewOptions() *Opts {
	return &Opts{
		DryRun:  false,
		Healthz: false,
		Port:    443,
	}
}
