/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/**
* Package main 整个工程的主入口
 */
package main

import (
	"fmt"
	"time"

	"openfuyao.com/kae-device-plugin/pkg/deviceplugin"
	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	pluginEndpointPrefix = "hisi"
	maxFailTimes         = 3
	fiveSeconds          = 5
	tenMinutes           = 1
)

var (
	retryTime = fiveSeconds * time.Second
	waitTime  = tenMinutes * time.Minute
)

func main() {
	defer zlog.Sync()
	zlog.Info("kae device plugin start")
	kaeManager := deviceplugin.NewKAEManager()
	deviceplugin.RemoveKAESock()
	deviceplugin.SetUname()
	deviceplugin.SetMod()
	var failTimes int
	for {
		err := kaeManager.Start()
		if err == nil {
			break
		}
		failTimes++
		// 重试三次后，1min 后重试
		if failTimes == maxFailTimes {
			zlog.Errorf("start device plugin failed, try again latter...")
			time.Sleep(waitTime)
			failTimes = 0
		}
		zlog.Errorf("deviceplugin.Start() failed: %v", err)
		time.Sleep(retryTime)
	}
	kaeManager.Serve(fmt.Sprintf("%s-%d.sock", pluginEndpointPrefix, time.Now().Unix()))
}
