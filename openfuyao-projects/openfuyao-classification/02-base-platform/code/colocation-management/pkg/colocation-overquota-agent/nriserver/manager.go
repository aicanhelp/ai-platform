/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package nriserver

import (
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-overquota-agent/apps"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/config"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks/actions/beresource"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks/actions/cpuset"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks/actions/gpu"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks/actions/numaaffinity"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/server"
	"openfuyao.com/colocation-management/pkg/common"
)

// Manager nriManager interface
type Manager interface {
	Setup()
}

// NriManager nriManager
type NriManager struct {
	config     config.Config
	nriServer  *server.NriServer
	kubeClient *kubernetes.Clientset
}

// NewNriManager 创建nriManager
func NewNriManager(cfg *apps.Configuration) (NriManager, error) {
	nriserver, err := server.NewNriServer()
	if err != nil {
		klog.Errorf("new server server error %v", err)
		return NriManager{}, err
	}

	baseConfig := config.Config{
		CgroupRoot:   cfg.CgroupRoot,
		CgroupDriver: cfg.CgroupDriver,
		NumaAffinity: config.NumaAffinityConfig{
			Enabled: cfg.NumaAffinityEnabled,
		},
	}

	k8sConfig, err := rest.InClusterConfig()
	var clientset *kubernetes.Clientset
	if err != nil {
		klog.V(common.AdvanceDebugLog).Infof(
			"Failed to get in-cluster config: %v. NUMA affinity ConfigMap reading will be disabled.", err)
	} else {
		clientset, err = kubernetes.NewForConfig(k8sConfig)
		if err != nil {
			klog.V(common.AdvanceDebugLog).Infof(
				"Failed to create Kubernetes clientset: %v. NUMA affinity ConfigMap reading will be disabled.", err)
		}
	}

	return NriManager{
		config:     baseConfig,
		nriServer:  nriserver,
		kubeClient: clientset,
	}, nil
}

// Setup 启动nriServer
func (nm *NriManager) Setup() {
	klog.Infof("server is starting...")

	nm.ValidConfig()
	klog.Infof("Registering standard hook actions...")
	beresource.NewBEResourceRunner().RegisterFunc()
	cpuset.NewCpuSetRunner().RegisterFunc()
	gpu.NewGpuRunner().RegisterFunc()

	numaAffinityRunner := numaaffinity.GetNUMAAffinityRunner()
	if err := numaAffinityRunner.Init(nm.config); err != nil {
		klog.Errorf("Failed to initialize NUMA affinity runner: %v", err)
	}

	numaAffinityRunner.RegisterFunc()
	klog.Infof("All hook actions registered. Starting NRI server...")
	go nm.nriServer.Start()
}

// ValidConfig 检验配置
func (nm *NriManager) ValidConfig() {
	if nm.kubeClient == nil {
		return
	}

	switch nm.config.CgroupDriver {
	case "systemd", "cgroupfs":
		core.SetCgroupDriver(nm.config.CgroupDriver)
	default:
		return
	}
}

// Shutdown 关闭nriServer
func (nm *NriManager) Shutdown() {
	klog.Infof("server is shutdown...")
}
