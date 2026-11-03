/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package main starts the UB network bandwidth operator process.
package main

import (
	"fmt"
	"os"
	"syscall"
	"time"

	"github.com/urfave/cli/v2"
	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/operator"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/reconciler"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/watch"
)

const restartDelay = 30 * time.Second

func main() {
	app := cli.NewApp()
	app.Name = "UB Network Bandwidth Operator"
	app.Usage = "UB Network Bandwidth QoS Operator for Kubernetes"
	app.Action = func(c *cli.Context) error {
		return start(c)
	}

	var configPath string
	var healthPort string

	app.Flags = []cli.Flag{
		&cli.StringFlag{
			Name:        "config-path",
			Value:       "/etc/ub-network-bandwidth/config.yaml",
			Usage:       "ConfigMap mounted config file path",
			Destination: &configPath,
			EnvVars:     []string{"CONFIG_PATH"},
		},
		&cli.StringFlag{
			Name:        "health-port",
			Value:       "8089",
			Usage:       "Health check HTTP port",
			Destination: &healthPort,
			EnvVars:     []string{"HEALTH_PORT"},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		klog.Error(err)
		os.Exit(1)
	}
}

func start(c *cli.Context) error {
	configPath := c.String("config-path")
	healthPort := c.String("health-port")

	klog.Infof("starting %s", c.App.Name)
	klog.Infof("config path: %s", configPath)
	klog.Infof("health port: %s", healthPort)

	klog.Infof("starting health server on port %s", healthPort)
	operator.StartHealthServer(healthPort)

	klog.Info("starting OS signal watcher")
	sigs := watch.Signals(syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
	if sigs == nil {
		return fmt.Errorf("signal watcher is not initialized")
	}

	manager := reconciler.NewBandwidthManager()

	var o operator.Interface

	for {
		if o != nil {
			o.Stop()
		}

		klog.Info("creating bandwidth operator")
		var err error
		o, err = operator.New(
			operator.WithManager(manager),
			operator.WithConfigPath(configPath),
			operator.WithHealthPort(healthPort),
		)
		if err != nil {
			return fmt.Errorf("error creating operator: %w", err)
		}

		if err := o.Start(); err != nil {
			klog.Errorf("error starting operator: %v, retrying in 30s...", err)
			if waitBeforeRestart(sigs, o, restartDelay) {
				continue
			}
			return nil
		}

		if waitForOperatorEvent(o, sigs) {
			continue
		}
		return nil
	}
}

func waitBeforeRestart(sigs <-chan os.Signal, o operator.Interface, delay time.Duration) bool {
	if sigs == nil {
		klog.Error("signal watcher is not initialized")
		stopOperator(o)
		return false
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-timer.C:
		return true
	case s := <-sigs:
		return handleSignal(s, o)
	}
}

func waitForOperatorEvent(o operator.Interface, sigs <-chan os.Signal) bool {
	if sigs == nil {
		klog.Error("signal watcher is not initialized")
		stopOperator(o)
		return false
	}
	if o == nil {
		klog.Error("operator is not initialized")
		return false
	}

	done := o.Done()
	if done == nil {
		klog.Error("operator done channel is not initialized")
		stopOperator(o)
		return false
	}

	select {
	case <-done:
		klog.Warning("operator exited unexpectedly, restarting")
		return true
	case s := <-sigs:
		return handleSignal(s, o)
	}
}

func handleSignal(s os.Signal, o operator.Interface) bool {
	switch s {
	case syscall.SIGHUP:
		klog.Info("received SIGHUP, restarting")
		return true
	default:
		klog.Infof("received signal %v, shutting down", s)
		stopOperator(o)
		return false
	}
}

func stopOperator(o operator.Interface) {
	if o != nil {
		o.Stop()
	}
}
