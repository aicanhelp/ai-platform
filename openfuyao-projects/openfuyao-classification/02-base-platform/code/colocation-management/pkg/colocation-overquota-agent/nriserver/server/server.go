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

// Package server 提供nriServer
package server

import (
	"context"
	"fmt"
	"time"

	"github.com/containerd/nri/pkg/api"
	"github.com/containerd/nri/pkg/stub"
	"github.com/google/cadvisor/utils"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
)

const (
	nriSocketPath = "/host-run/nri/nri.sock"
	events        = "CreateContainer,UpdateContainer,RunPodSandbox,RemovePodSandbox"
	pluginName    = "colocation-overquota-agent"
	pluginIndex   = "10"

	maxRetrySteps  = 999
	baseRetryDelay = 3 * time.Second
	retryFactor    = 1.0
	retryJitter    = 0.1
)

// NriServer  is the nri server
type NriServer struct {
	stub         stub.Stub
	eventMask    stub.EventMask
	options      []stub.Option // server stub options
	conf         configuration
	finishedSync chan struct{}
}

type configuration struct {
	PluginName  string
	PluginIndex string
}

// NewNriServer create nriServer
func NewNriServer() (*NriServer, error) {

	if !utils.FileExists(nriSocketPath) {
		klog.Errorf("server plugin socket does not exist %v", nriSocketPath)
		// 如果nriSocket不存在 返回错误 退出当前进程
		return nil, fmt.Errorf("server plugin socket does not exist %v", nriSocketPath)
	}

	options := []stub.Option{
		stub.WithPluginName(pluginName),
		stub.WithPluginIdx(pluginIndex),
		stub.WithSocketPath(nriSocketPath),
	}

	mask, err := api.ParseEventMask(events)
	if err != nil {
		klog.Errorf("server plugin parse eventMask error %v ", err)
		return nil, err
	}

	plugin := &NriServer{
		options: options,
		conf: configuration{
			pluginName,
			pluginIndex,
		},
		eventMask: mask,
	}

	plugin.stub, err = stub.New(plugin, append(options, stub.WithOnClose(plugin.onClose))...)
	if err != nil {
		klog.Errorf("server plugin create stub error %v ", err)
	}

	return plugin, nil
}

func (plugin *NriServer) onClose() {
	// restart
	klog.Infof("server plugin restart")
	f := func() bool {
		//  reconnect
		stub, err := stub.New(plugin, append(plugin.options, stub.WithOnClose(plugin.onClose))...)
		if err != nil {
			klog.Errorf("server plugin restart stub error %v ", err)
			return false
		}
		plugin.stub = stub
		// Start需要检查 否则重启后依然无法建立连接
		err = plugin.Start()
		if err != nil {
			klog.Errorf("server plugin restart  error %v ", err)
			return false
		}
		return true
	}
	err := plugin.RetryOnClose(f, func() error {
		return nil
	})
	if err != nil {
		klog.Errorf("server plugin restart failed")
	}
}

// Configure NriServer
func (plugin *NriServer) Configure(context.Context, string, string, string) (api.EventMask, error) {
	var eventMask api.EventMask
	eventMask, err := api.ParseEventMask(events)
	if err != nil {
		klog.Errorf("server plugin parse eventMask error %v ", err)
		return 0, err
	}
	plugin.eventMask = eventMask

	return plugin.eventMask, nil
}

// Start NriServer
func (plugin *NriServer) Start() error {
	return plugin.stub.Run(context.Background())
}

// WaitReady wait sync event finish
func (plugin *NriServer) WaitReady() {
	// <-plugin.finishedSync
}

// Synchronize NriServer
func (plugin NriServer) Synchronize(ctx context.Context, pods []*api.PodSandbox,
	containers []*api.Container) ([]*api.ContainerUpdate, error) {
	// 添加 sync all container
	// notify service handler to start
	// socket重启时 不能对已经关闭对channel再次关闭
	return nil, nil
}

// Stop NriServer
func (p *NriServer) Stop() {
	p.stub.Stop()
}

// CreateContainer is called when a container is created.
func (plugin *NriServer) CreateContainer(ctx context.Context, pod *api.PodSandbox,
	container *api.Container) (*api.ContainerAdjustment, []*api.ContainerUpdate, error) {
	// Convert NRIRawPod2ContainerInfo
	containerInfo := &core.ContainerInfo{
		ContainerRequest:   core.ContainerRequest{},
		ContainerResponese: core.ContainerResponse{},
	}
	containerInfo.ConvertNRIRawPod2ContainerInfo(pod, container)

	// CreateContainer hook
	err := hooks.RunHook(common.PreCreateContainer, containerInfo)
	if err != nil {
		return nil, nil, err
	}
	if plugin.canRunUpdate(containerInfo.Admit()) {
		// update to cgroup
		adjust, _, err := containerInfo.UpdateContainer()
		if err != nil {
			containerInfo.Deny()
			return nil, nil, err
		}
		klog.Infof("CreateContainer server run success %s/%s", pod.GetNamespace(), container.GetName())

		return adjust, nil, nil
	}
	return nil, nil, fmt.Errorf("admit false %v", containerInfo.Admit().Reason)
}

// RunPodSandbox is called when a pod sandbox is created.
func (plugin *NriServer) RunPodSandbox(ctx context.Context, pod *api.PodSandbox) error {
	podInfo := &core.PodInfo{}
	podInfo.ConvertNRIRawPod2PodInfo(pod)
	// PreRunPodSandbox fn(data)
	// RunPodSandBox hook
	err := hooks.RunHook(common.PreRunPodSandbox, podInfo)
	if err != nil {
		return err
	}

	if plugin.canRunUpdate(podInfo.Admit()) {
		// update to cgroup
		podInfo.Update()
		// croup update
		klog.Infof("RunPodSandBox server run success %s/%s", pod.GetNamespace(), pod.GetName())
		return nil
	}
	return fmt.Errorf("admit false %v", podInfo.Admit().Reason)

}

// UpdateContainer is called when a container is updated.
func (plugin *NriServer) UpdateContainer(ctx context.Context, pod *api.PodSandbox,
	container *api.Container, r *api.LinuxResources) ([]*api.ContainerUpdate, error) {
	// Convert NRIRawPod2ContainerInfo
	containerInfo := &core.ContainerInfo{
		ContainerRequest:   core.ContainerRequest{},
		ContainerResponese: core.ContainerResponse{},
	}
	containerInfo.ConvertNRIRawPod2ContainerInfo(pod, container)

	// UpdateContainer hook
	err := hooks.RunHook(common.PreUpdateContainerResources, containerInfo)
	if err != nil {
		return nil, err
	}

	if plugin.canRunUpdate(containerInfo.Admit()) {
		// update to cgroup
		_, update, err := containerInfo.UpdateContainer()
		if err != nil {
			containerInfo.Deny()
			return nil, err
		}

		klog.Infof("UpdateContainer server run success %s/%s", pod.GetNamespace(), container.GetName())
		return []*api.ContainerUpdate{update}, nil
	}
	return nil, fmt.Errorf("admit false %v", containerInfo.Admit().Reason)
}

// RemovePodSandbox is called when a pod sandbox is removed.
func (plugin *NriServer) RemovePodSandbox(ctx context.Context, pod *api.PodSandbox) error {
	podInfo := &core.PodInfo{}
	podInfo.ConvertNRIRawPod2PodInfo(pod)

	// RemovePodSandbox hook
	err := hooks.RunHook(common.PreRemoveRunPodSandbox, podInfo)
	if err != nil {
		return err
	}
	if plugin.canRunUpdate(podInfo.Admit()) {
		// update to cgroup
		podInfo.Update()
		klog.Infof("RemovePodSandbox server run success %s/%s", pod.GetNamespace(), pod.GetName())
		return nil
	}

	return fmt.Errorf("admit false %v", podInfo.Admit().Reason)
}

func (plugin *NriServer) canRunUpdate(result core.AdmitResult) bool {
	if result.Admit {
		return true
	} else {
		return false
	}
}

// RetryOnClose retry on close
func (plugin *NriServer) RetryOnClose(retryFunc func() bool, eroorFunc func() error) error {

	// 添加可配置
	var DefaultRetry = wait.Backoff{
		Steps:    maxRetrySteps,
		Duration: baseRetryDelay,
		Factor:   retryFactor,
		Jitter:   retryJitter,
	}
	err := wait.ExponentialBackoff(DefaultRetry, func() (bool, error) {
		return retryFunc(), eroorFunc()
	})
	if err == wait.ErrWaitTimeout {
		return fmt.Errorf("retry on close timeout")
	}
	return err
}
