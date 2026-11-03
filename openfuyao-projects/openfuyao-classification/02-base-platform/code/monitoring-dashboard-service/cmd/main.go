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

// Package main is the entry of monitoring monitor service
package main

import (
	"context"
	"fmt"

	"monitoring-dashboard-service/pkg/config/mdserver"
	"monitoring-dashboard-service/pkg/config/runtime"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

func main() {
	zlog.LogInfo("monitoring monitor service start")

	// create http server
	runOptions := runtime.NewRunConfig()

	ctx := context.TODO()

	apiServer, err := mdserver.NewAPIServer(runOptions)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError build apiServer: %s", err)
		zlog.LogError(errorMessage)
	}

	err = apiServer.Run(ctx)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError running apiServer: %s", err)
		zlog.LogError(errorMessage)
	}
}
