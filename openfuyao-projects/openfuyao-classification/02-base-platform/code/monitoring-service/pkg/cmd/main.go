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

// Package main serves as the entry point for the monitoring service application, designed
// to provide a comprehensive solution for monitoring resources and activities within a
// Kubernetes environment. This package leverages internal APIs and server configurations
// to instantiate and manage a monitoring server, which interfaces with Prometheus
// to aggregate and query monitoring data.
package main

import (
	"context"

	"openfuyao.com/monitoring-service/pkg/api/monitoring"
	"openfuyao.com/monitoring-service/pkg/cmd/config"
	"openfuyao.com/monitoring-service/pkg/zlog"
)

func main() {
	defer zlog.Sync()
	zlog.Info("hello openfuyao")

	runOptions := config.NewRunConfig()
	ctx := context.TODO()
	apiServer, err := monitoring.NewAPIServer(runOptions, ctx)

	if err != nil {
		zlog.FormatFatal("Failed to NewAPIServer: %v", err)
	}
	err = apiServer.Run(ctx)
	if err != nil {
		zlog.FormatFatal("Failed to Run APIServer: %v", err)
	}
}
