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

// Package main serves as the entry point for the volcano config service
package main

import (
	"context"
	"os"

	"volcano-config-service/pkg/api/numaexporter"
	"volcano-config-service/pkg/cmd/config"
	"volcano-config-service/pkg/controller"
	"volcano-config-service/pkg/server"
	"volcano-config-service/pkg/zlog"
)

func main() {
	// 获取运行模式环境变量
	runMode := os.Getenv("RUN_MODE")

	// 根据运行模式选择启动的组件
	if runMode == "DEPLOYMENT" || runMode == "" { // 默认为 Deployment 模式
		// 1. 启动 NUMA 数据上报服务 (两种模式都需要)
		go numaexporter.InitServer()

		// 2. 创建运行配置
		runOptions := config.NewRunConfig()
		if errs := runOptions.Validate(); len(errs) != 0 {
			zlog.FormatFatal("Failed to Validate RunConfig: %v", errs)
		}

		// 3. 创建并启动 HTTP server
		ctx := context.TODO()
		httpServer, err := server.NewServer(runOptions, ctx)
		if err != nil {
			zlog.FormatFatal("Failed to NewServer: %v", err)
		}

		zlog.FormatInfo("Run Http Server")
		err = httpServer.Run(ctx)
		if err != nil {
			zlog.FormatFatal("Failed to Run server: %v", err)
		}
	} else if runMode == "DAEMONSET" {
		// 4. 初始化和启动 controller manager
		mgr, err := controller.InitControllerManager()
		if err != nil {
			zlog.FormatFatal("Failed to init controller manager: %v", err)
		}

		zlog.FormatInfo("Starting Controller Manager")
		if err := controller.StartManager(mgr); err != nil {
			zlog.FormatFatal("Failed to start controller manager: %v", err)
		}

		// 添加阻塞逻辑
		stop := make(chan struct{})
		<-stop // 永久阻塞，直到进程被终止
	} else {
		zlog.FormatFatal("Unknown RUN_MODE: %s. Must be either DEPLOYMENT or DAEMONSET", runMode)
	}
}
