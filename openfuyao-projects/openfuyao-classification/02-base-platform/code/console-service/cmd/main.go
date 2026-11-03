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

package main

import (
	"context"

	"console-service/cmd/config"
	"console-service/pkg/server"
	"console-service/pkg/zlog"
)

func main() {
	defer zlog.Sync()
	zlog.Info("console-serivce start")

	// 创建http server、各种资源操作的配置对象，目前只实现了k8s的配置对象
	runOptions := config.NewRunConfig()
	// 校验server和k8s配置
	if errs := runOptions.Validate(); len(errs) != 0 {
		zlog.Fatalf("Failed to Validate RunConfig: %v", errs)
	}

	ctx := context.TODO()
	consoleServer, err := server.NewServer(runOptions, ctx)
	if err != nil {
		zlog.Fatalf("Failed to NewServer: %v", err)
	}

	err = consoleServer.Run(ctx)
	if err != nil {
		zlog.Fatalf("Failed to Run cServer: %v", err)
	}

}
