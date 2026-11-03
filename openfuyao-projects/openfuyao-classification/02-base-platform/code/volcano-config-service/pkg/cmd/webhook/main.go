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

// Package main contains the entry point for all functions
package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"volcano-config-service/pkg/cmd/webhook/options"
	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/webhook"
	"volcano-config-service/pkg/zlog"
)

func main() {

	zlog.Info("numa-affinity-webhook start!")

	newOptions := options.NewOptions()
	options.Flags(newOptions)

	pair, err := tls.LoadX509KeyPair(newOptions.CertFile, newOptions.KeyFile)
	if err != nil {
		zlog.Errorf("load key pair Failed: %v", err)
		return
	}

	webhookServer := &webhook.WhServer{
		Server: &http.Server{
			Addr:      fmt.Sprintf(":%v", newOptions.Port),
			TLSConfig: &tls.Config{Certificates: []tls.Certificate{pair}},
		},
	}

	// 定义入口
	mux := http.NewServeMux()
	mux.HandleFunc("/mutate", webhookServer.Serve)

	webhookServer.Server.Handler = mux

	// 开启协程进行处理
	go func() {
		if err := webhookServer.Server.ListenAndServeTLS("", ""); err != nil {
			// 此时服务中止
			zlog.Errorf("listen and serve failed, nriServer webhook: %v,please connect admin", err)
		}
	}()

	// 给服务器一点时间启动
	time.Sleep(constant.TimeStart * time.Second)

	zlog.Infof("检查webhook service binding情况")
	err = webhook.WaitForSvcRunning(time.Second*constant.TimeInterval, time.Minute*constant.TimeDuration,
		newOptions.SvcNamespace, newOptions.SvcName)
	if err != nil {
		zlog.Errorf("wait for service running failed, %v", err)
		return
	}

	// 监听OS信号
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)
	<-signalChan

	zlog.Infof("Got OS shutdown signal, shutting down webhook nriServer gracefully...")
	webhookServer.Server.Shutdown(context.Background())

}
