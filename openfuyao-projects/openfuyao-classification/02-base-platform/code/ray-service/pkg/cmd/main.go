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

// Package main serves as the entry point for the raycluster config service
package main

import (
	"context"

	"openfuyao-rayservice/pkg/cmd/config"
	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/database"
	"openfuyao-rayservice/pkg/server"
	"openfuyao-rayservice/pkg/zlog"
)

func main() {
	defer zlog.Sync()
	zlog.Info("openfuyao-rayservice start")

	// 创建http server、各种资源操作的配置对象，目前只实现了k8s的配置对象
	runOptions := config.NewRunConfig()
	// 校验server和k8s配置
	if errs := runOptions.Validate(); len(errs) != 0 {
		zlog.FormatFatal("Failed to Validate RunConfig: %v", errs)
	}

	ctx := context.Background()
	dbConnection := database.Connect(runOptions)
	ctx = context.WithValue(ctx, constant.ContextDBKey, dbConnection)
	Server, err := server.NewServer(runOptions, ctx)
	if err != nil {
		zlog.FormatFatal("Failed to NewServer: %v", err)
	}

	err = Server.Run(ctx)
	if err != nil {
		zlog.FormatFatal("Failed to Run cServer: %v", err)
	}
}
