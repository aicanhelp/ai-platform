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
	"flag"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"syscall"

	"github.com/fsnotify/fsnotify"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"huawei.com/vxpu-device-plugin/pkg/api/runtime/service"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
	"huawei.com/vxpu-device-plugin/watchers"
)

const (
	xpuSockPath             = "xpu.sock"
	defaultDeviceSplitCount = 2
	defaultLogDir           = "/var/log/xpu/xpu-device-plugin"
)

var (
	resourceName string
)

func events(watcher *fsnotify.Watcher, sigs chan os.Signal, pluginInst *plugin.DevicePlugin) bool {
	for {
		select {
		case event := <-watcher.Events:
			if event.Name == v1beta1.KubeletSocket && event.Op&fsnotify.Create == fsnotify.Create {
				log.Infof("inotify: %s created, restarting.", v1beta1.KubeletSocket)
				return true
			}
		case err := <-watcher.Errors:
			log.Infof("inotify: %s", err)
		case s := <-sigs:
			switch s {
			case syscall.SIGHUP:
				log.Infoln("Received SIGHUP, restarting.")
				return true
			default:
				log.Infof("Received signal %v, shutting down.", s)
				pluginInst.Stop()
				return false
			}
		}
	}
}

func start() error {
	syscall.Umask(0)

	logFileName := path.Join(config.LogDir, "xpu-device-plugin.log")
	if err := log.InitLogging(logFileName); err != nil {
		fmt.Fprintf(os.Stderr, "failed to init logging: %v\n", err)
		os.Exit(1)
	}

	if err := xpu.Init(); err != nil {
		log.Errorf("xpu Init failed: %v", err)
		return err
	}
	defer xpu.Uninit()

	log.Infoln("Starting FS watcher.")
	watcher, err := watchers.NewFSWatcher(v1beta1.DevicePluginPath)
	if err != nil {
		return fmt.Errorf("failed to create FS watcher: %v", err)
	}
	defer watcher.Close()

	log.Infoln("Starting OS watcher.")
	sigs := watchers.NewOSWatcher(syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	util.SetNodeConfig()

	cache := plugin.NewDeviceCache()
	cache.Start()
	defer cache.Stop()

	register := plugin.NewDeviceRegister(cache)
	register.Start()

	service.Start()

	pluginInst := plugin.NewDevicePlugin(
		resourceName,
		cache,
		filepath.Clean(filepath.Join(v1beta1.DevicePluginPath, xpuSockPath)))

	if len(pluginInst.Devices()) == 0 {
		return fmt.Errorf("there are no devices to serve for current node")
	}

	for {
		if err := pluginInst.Start(); err != nil {
			log.Errorln("Start vxpu device plugin failed!")
			return err
		}

		if restart := events(watcher, sigs, pluginInst); restart != true {
			break
		}
	}
	return nil
}

func main() {
	flag.UintVar(&config.DeviceSplitCount, "device-split-count",
		defaultDeviceSplitCount, "the number for NVIDIA device split")
	flag.StringVar(&config.NodeName, "node-name", os.Getenv("NODE_NAME"), "node name")
	flag.StringVar(&config.LogDir, "log-dir", defaultLogDir, "log storage directory")
	flag.StringVar(&resourceName, "resource-name", xpu.VxpuNumber, "resource name")
	flag.Parse()
	if err := start(); err != nil {
		log.Fatalln(err)
	}
}
