/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/klogr"
	"sigs.k8s.io/controller-runtime"

	"openfuyao.com/colocation-management/cmd/colocation-overquota-agent/apps"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/report"
	"openfuyao.com/colocation-management/pkg/common"
)

const (
	// numberOfBlockingGoroutines is the number of blocking goroutines.
	numberOfBlockingGoroutines = 1

	// gracefulShutdownTimeout is the timeout for graceful shutdown.
	gracefulShutdownTimeout = 30 * time.Second
)

func main() {
	config := apps.NewConfiguration()
	config.InitFlags(flag.CommandLine)
	klog.InitFlags(nil)
	flag.Parse()
	rand.Seed(time.Now().UnixNano())
	controllerruntime.SetLogger(klogr.New())
	config.PrintAndExitIfRequested()

	go wait.Forever(klog.Flush, common.KlogFlushInterval)
	defer klog.Flush()

	if config.EnablePprof {
		go func() {
			if err := http.ListenAndServe(config.PprofAddr, nil); err != nil {
				klog.Fatalf("unable to start pprof. %v", err)
			}
		}()
	}

	mgrCfg := controllerruntime.GetConfigOrDie()

	ctx, errorChan := SetupSignalHandler()

	reporter, err := report.NewMetricsReporter(ctx, mgrCfg, config)
	if err != nil {
		klog.Error("create metrics reporter failed. %v", err)
		os.Exit(1)
	}

	nriManager, err := nriserver.NewNriManager(config)
	if err != nil {
		klog.Error("create nri server failed. %v", err)
		os.Exit(1)
	}

	klog.Info("colocation overquota agent starting")
	var wg sync.WaitGroup
	wg.Add(numberOfBlockingGoroutines)
	go func() {
		defer wg.Done()
		nriManager.Setup()
	}()

	wg.Wait()
	klog.Info("All tasks completed, waiting for shutdown signal...")
	select {
	case err := <-errorChan:
		klog.Errorf("Received shutdown signal: %v", err)
		os.Exit(1)
	case <-ctx.Done():
		klog.Info("Shutdown signal received, exiting...")
		runShutdownHander(ctx, reporter)
		os.Exit(0)
	}
}

func runShutdownHander(ctx context.Context, reporter *report.MetricsReporter) {
	klog.Infof("Shutting down  server...")
	reporter.Shutdown()
}

// SetupSignalHandler returns a context that will be cancelled when a
func SetupSignalHandler() (context.Context, <-chan error) {
	ctx, cancel := context.WithCancel(context.Background())
	errorChan := make(chan error, 1)

	c := make(chan os.Signal, 2)
	signal.Notify(c, os.Interrupt, os.Kill, syscall.SIGTERM, syscall.SIGQUIT)

	go func() {
		<-c
		cancel()
		klog.V(common.BasicDebugLog).Info("Received signal, exiting...")
		klog.Flush()
		select {
		case <-c:
			klog.Info("Received second signal, forcing immediate shutdown")
			errorChan <- fmt.Errorf("forced shutdown by second signal")
		case <-time.After(gracefulShutdownTimeout): // 添加超时保护
			klog.Info("Graceful shutdown timeout")
			errorChan <- fmt.Errorf("graceful shutdown timeout")
		}
	}()
	return ctx, errorChan
}
