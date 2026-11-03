// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package main

import (
	"fmt"
	"os"
	"time"

	"github.com/urfave/cli/v2"
	"k8s.io/klog/v2"

	"openfuyao.com/npu-feature-discovery/internal/lm"
)

const (
	sleepIntervalFlag = "sleep-interval"
	envSleepInterval  = "GFD_SLEEP_INTERVAL"
	sleepTime         = 60
)

func main() {
	app := setupCLIApp()

	if err := app.Run(os.Args); err != nil {
		klog.Error(err)
		os.Exit(1)
	}
}

// setupCLIApp 初始化CLI应用配置
func setupCLIApp() *cli.App {
	app := cli.NewApp()
	app.Name = "NPU Feature Discovery"
	app.Usage = "generate labels for NPU devices"
	app.Action = start
	app.Flags = []cli.Flag{
		&cli.DurationFlag{
			Name:    sleepIntervalFlag,
			Value:   sleepTime * time.Second,
			Usage:   "Time to sleep between labeling",
			EnvVars: []string{envSleepInterval},
		},
	}
	return app
}

func start(ctx *cli.Context) error {
	defer klog.Info("Exiting")

	sleepInterval := ctx.Duration(sleepIntervalFlag)

	for {
		klog.Info("Start running")
		restart, err := processLabels(ctx, sleepInterval)
		if err != nil {
			return err
		}
		if !restart {
			return nil
		}
	}
}

// processLabels 处理标签生成的核心逻辑
func processLabels(ctx *cli.Context, sleepInterval time.Duration) (bool, error) {
	// 创建标签生成器
	labelers, err := lm.NewLabelers(ctx)
	if err != nil {
		return false, fmt.Errorf("failed to create labelers: %w", err)
	}

	// 获取标签
	labels := labelers.Labels()

	// 处理空标签情况
	if len(labels) == 0 {
		klog.Warning("No labels generated from any source")
		return true, nil
	}

	// 输出标签
	klog.Info("Creating Labels")
	if err := lm.Output(ctx, labels); err != nil {
		return false, err
	}

	// 等待下一轮执行
	klog.Info("Sleeping for ", sleepInterval)
	time.Sleep(sleepInterval)
	return true, nil
}
