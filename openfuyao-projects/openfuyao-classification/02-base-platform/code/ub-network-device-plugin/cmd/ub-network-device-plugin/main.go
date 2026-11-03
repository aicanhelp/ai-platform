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

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"

	ubse "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/engine"
	"github.com/fsnotify/fsnotify"
	"github.com/urfave/cli/v2"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"gitcode.com/openFuyao/ub-network-device-plugin/plugin"
	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
	"gitcode.com/openFuyao/ub-network-device-plugin/watch"
)

func main() {
	app := cli.NewApp()
	o := &plugin.StartOpts{}
	app.Name = "UB Network Device Plugin"
	app.Usage = "UB Network Device Plugin for Kubernetes"
	app.Action = func(c *cli.Context) error {
		return start(c, o)
	}

	app.Flags = []cli.Flag{
		&cli.StringFlag{
			Name:        "kubelet-socket",
			Value:       pluginapi.KubeletSocket,
			Usage:       "Kubelet's IPC socket path",
			Destination: &o.KubeletSocket,
			EnvVars:     []string{"KUBELET_SOCKET"},
		},
		&cli.DurationFlag{
			Name:        "sync-interval",
			Aliases:     []string{"s"},
			Value:       30 * time.Second,
			Usage:       "Interval in seconds for syncing device states (default: 30 seconds)",
			Destination: &o.SyncInterval,
		},
		&cli.StringFlag{
			Name:        "health-port",
			Value:       "8088",
			Usage:       "Health check port",
			Destination: &o.HealthPort,
		},
	}
	o.Flags = app.Flags

	err := app.Run(os.Args)
	if err != nil {
		klog.Error(err)
		os.Exit(1)
	}
}

func start(c *cli.Context, o *plugin.StartOpts) error {
	klog.Infof("starting %s", c.App.Name)

	klog.Infof("Starting health server on port %s", o.HealthPort)
	plugin.StartHealthServer(o.HealthPort)

	kubeletSocketDir := filepath.Dir(o.KubeletSocket)
	klog.Infof("Starting FS watcher for %v", kubeletSocketDir)
	watcher, err := watch.Files(kubeletSocketDir)
	if err != nil {
		return fmt.Errorf("failed to create FS watcher for %s: %v", pluginapi.DevicePluginPath, err)
	}
	defer watcher.Close()

	klog.Info("Starting OS watcher")
	sigs := watch.Signals(syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	if err = utils.EnsureDevicePermission(utils.TidDevicePath, utils.Permission); err != nil {
		return fmt.Errorf("error setting device permission: %v", err)
	}

	_, err = ubse.UbsEngineClientInitialize("")
	if err != nil {
		return fmt.Errorf("error initializing UBS engine: %v", err)
	}

	var started bool
	var restartTimeout <-chan time.Time
	var plugins []plugin.Interface
restart:
	if started {
		err := stopPlugins(plugins)
		if err != nil {
			return fmt.Errorf("error stopping plugins from previous run: %v", err)
		}
	}

	klog.Info("Starting plugin.")
	plugins, restartPlugins, err := startPlugins(c, o)
	if err != nil {
		return fmt.Errorf("error starting plugins: %v", err)
	}
	started = true

	if restartPlugins {
		klog.Infof("Failed to start one or more plugins. Retrying in 30s...")
		restartTimeout = time.After(30 * time.Second)
	}

	for {
		select {
		case <-restartTimeout:
			goto restart

		case event := <-watcher.Events:
			if o.KubeletSocket != "" && event.Name == o.KubeletSocket && event.Op&fsnotify.Create == fsnotify.Create {
				klog.Infof("inotify: %s created, restarting.", o.KubeletSocket)
				goto restart
			}

		case <-watcher.Errors:
			klog.Infof("inotify: %s", err)

		case s := <-sigs:
			switch s {
			case syscall.SIGHUP:
				klog.Info("Received SIGHUP, restarting.")
				goto restart
			default:
				klog.Infof(`Received signal "%v", shutting down.`, s)
				goto exit
			}
		}
	}

exit:
	err = stopPlugins(plugins)
	if err != nil {
		return fmt.Errorf("error stopping plugins: %v", err)
	}
	ubse.UbsEngineClientFinalize()
	return nil
}

func startPlugins(c *cli.Context, o *plugin.StartOpts) ([]plugin.Interface, bool, error) {
	resources, err := resource.InitDefaultResources()
	if err != nil {
		return nil, false, fmt.Errorf("error initializing resources: %v", err)
	}

	klog.Info("Retrieving plugins.")
	plugins, err := getPlugins(c.Context, resources)
	if len(plugins) == 0 || err != nil {
		return nil, false, fmt.Errorf("error getting plugins: %v", err)
	}

	// plugin start
	started := 0
	for _, p := range plugins {
		if len(p.Devices()) == 0 {
			continue
		}

		err := p.Start(o)
		if err != nil {
			klog.Errorf("Error starting plugin: %v", err)
			return plugins, true, nil
		}
		started++
	}

	if started == 0 {
		return plugins, false, errors.New("no plugins started")
	}

	return plugins, false, nil
}

func stopPlugins(plugins []plugin.Interface) error {
	klog.Info("Stopping plugin.")
	var errs error
	for _, p := range plugins {
		errs = errors.Join(errs, p.Stop())
	}
	return errs
}

func getPlugins(ctx context.Context, resources *resource.Resources) ([]plugin.Interface, error) {
	plugins, err := plugin.New(ctx, plugin.WithResources(resources))
	if err != nil {
		return nil, fmt.Errorf("error creating plugins: %v", err)
	}
	return plugins, nil
}
