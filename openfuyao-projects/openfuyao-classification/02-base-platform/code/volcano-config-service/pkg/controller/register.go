/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
// Package controller
package controller

import (
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"volcano-config-service/pkg/zlog"
)

var (
	scheme = runtime.NewScheme()
)

// KubeletConfigPath 存储 kubelet 配置文件路径
var KubeletConfigPath string

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(addToScheme(scheme))

	// 初始化 kubelet 配置文件路径
	possiblePaths := []string{
		"/etc/kubernetes/kubelet-config.yaml",
		"/var/lib/kubelet/config.yaml",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			KubeletConfigPath = path
			zlog.Infof("Found kubelet config at: %s", KubeletConfigPath)
			return
		}
	}

	// 如果都没找到，设置默认值
	zlog.Warn("Kubelet config not found in any of the expected locations, using default")
	KubeletConfigPath = "/var/lib/kubelet/config.yaml"
}

// InitControllerManager 创建并初始化 controller manager
func InitControllerManager() (manager.Manager, error) {
	metricsPort := os.Getenv("METRICS_PORT")
	if metricsPort == "" {
		metricsPort = "9180"
	}
	healthProbePort := os.Getenv("HEALTH_PROBE_PORT")
	if healthProbePort == "" {
		healthProbePort = "9181"
	}

	// 创建 controller manager
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     fmt.Sprintf(":%s", metricsPort),
		HealthProbeBindAddress: fmt.Sprintf(":%s", healthProbePort),
		LeaderElection:         false,
		LeaderElectionID:       "node-config-controller",
	})
	if err != nil {
		return nil, err
	}
	// 设置 NodeConfig controller
	if err := setupNodeConfigController(mgr); err != nil {
		return nil, err
	}

	// 设置健康检查
	if err := setupHealthChecks(mgr); err != nil {
		return nil, err
	}
	return mgr, nil
}

// setupNodeConfigController 设置 NodeConfig controller
func setupNodeConfigController(mgr manager.Manager) error {
	controller := &NodeConfigController{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		// 状态/spec 更新必须直连 apiserver 读取最新版本：
		// DaemonSet 模式下每个节点一个 controller 实例，各自 cache
		// 的 resourceVersion 滞后，从 cache 读取再提交会持续 409 冲突
		// 且重试无效，导致节点状态卡在 Configuring（issue 92）
		APIReader: mgr.GetAPIReader(),
	}
	if err := controller.SetupWithManager(mgr); err != nil {
		return err
	}
	return nil
}

// setupHealthChecks 设置健康检查
func setupHealthChecks(mgr manager.Manager) error {
	// 添加存活检查
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		return err
	}
	// 添加就绪检查
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		return err
	}
	return nil
}

// StartManager 启动 controller manager
func StartManager(mgr manager.Manager) error {
	// 启动 manager
	ctx := ctrl.SetupSignalHandler()
	// 在一个新的 goroutine 中启动 manager
	go func() {
		zlog.Info("Starting controller manager")
		if err := mgr.Start(ctx); err != nil {
			zlog.FormatFatal("Problem running manager: %v", err)
		}
	}()
	// 等待缓存同步
	if ok := mgr.GetCache().WaitForCacheSync(ctx); !ok {
		return fmt.Errorf("failed to sync cache")
	}
	// 初始化 NodeConfig CR
	if err := InitializeNodeConfig(mgr.GetClient(), mgr.GetAPIReader()); err != nil {
		return fmt.Errorf("failed to initialize NodeConfig CR: %v", err)
	}
	return nil
}
