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

// Package core
package core

import (
	"github.com/containerd/nri/pkg/api"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/runnermanager"
	"openfuyao.com/colocation-management/pkg/utils"
)

const (
	// DefaultCPURequestMilliCores is the default CPU request in milli cores
	DefaultCPURequestMilliCores = 1000
	// CPUSharesPerCore is the number of shares per core
	CPUSharesPerCore = 1024
)

// ContainerMetadata is the metadata of a container
type ContainerMetadata struct {
	Name    string
	ID      string
	Sandbox bool
}

// ContainerInfo is the info of a container
type ContainerInfo struct {
	ContainerRequest   ContainerRequest
	ContainerResponese ContainerResponse
}

// Admit is the admit
func (c *ContainerInfo) Admit() AdmitResult {
	return CheckAnnotationAdmit(c.ContainerRequest.PodSandboxConfig.Annotations)
}

// Update is the update
func (c *ContainerInfo) Update() {
	if c.ContainerResponese.Resources.CPUShares != nil {
		err := runnermanager.GetRunner().
			SetCpuShares(c.ContainerRequest.PodSandboxConfig.Linux.CgroupParent, c.ContainerResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu shares: %v", err)
		}
	}
	if c.ContainerResponese.Resources.CFSQuota != nil {
		err := runnermanager.GetRunner().
			SetCpuQuota(c.ContainerRequest.PodSandboxConfig.Linux.CgroupParent, c.ContainerResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu quota: %v", err)
		}
	}
	if c.ContainerResponese.Resources.CPUSet != nil {
		err := runnermanager.GetRunner().
			SetCpuSet(c.ContainerRequest.PodSandboxConfig.Linux.CgroupParent, c.ContainerResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu set: %v", err)
		}
	}
	if c.ContainerResponese.Resources.CPUSetMem != nil {
		err := runnermanager.GetRunner().
			SetCpuSetMem(c.ContainerRequest.PodSandboxConfig.Linux.CgroupParent, c.ContainerResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting cpu set mem: %v", err)
		}
	}
	if c.ContainerResponese.Resources.MemoryLimit != nil {
		err := runnermanager.GetRunner().
			SetMemoryLimit(c.ContainerRequest.PodSandboxConfig.Linux.CgroupParent, c.ContainerResponese.Resources)
		if err != nil {
			klog.Errorf("fail setting memory limit: %v", err)
		}
	}
}

// UpdateContainer is the update container
func (c *ContainerInfo) UpdateContainer() (*api.ContainerAdjustment, *api.ContainerUpdate, error) {
	adjust := &api.ContainerAdjustment{}
	update := &api.ContainerUpdate{}
	if c.ContainerResponese.Resources.CPUSet != nil {
		adjust.SetLinuxCPUSetCPUs(*c.ContainerResponese.Resources.CPUSet)
		update.SetLinuxCPUSetCPUs(*c.ContainerResponese.Resources.CPUSet)
	}

	if c.ContainerResponese.Resources.CPUSetMem != nil {
		adjust.SetLinuxCPUSetMems(*c.ContainerResponese.Resources.CPUSetMem)
		update.SetLinuxCPUSetMems(*c.ContainerResponese.Resources.CPUSetMem)
	}

	if c.ContainerResponese.Resources.CFSQuota != nil {
		adjust.SetLinuxCPUQuota(*c.ContainerResponese.Resources.CFSQuota)
		update.SetLinuxCPUQuota(*c.ContainerResponese.Resources.CFSQuota)
	}

	if c.ContainerResponese.Resources.CPUShares != nil {
		adjust.SetLinuxCPUShares(uint64(*c.ContainerResponese.Resources.CPUShares))
		update.SetLinuxCPUShares(uint64(*c.ContainerResponese.Resources.CPUShares))
	}

	if c.ContainerResponese.Resources.MemoryLimit != nil {
		adjust.SetLinuxMemoryLimit(*c.ContainerResponese.Resources.MemoryLimit)
		update.SetLinuxMemoryLimit(*c.ContainerResponese.Resources.MemoryLimit)
	}

	if c.ContainerResponese.AddContainerEnvs != nil {
		for k, v := range c.ContainerResponese.AddContainerEnvs {
			adjust.AddEnv(k, v)
		}
	}

	return adjust, update, nil

}

// Deny is the deny
func (c *ContainerInfo) Deny() AdmitResult {
	// to record reason and status
	return AdmitResult{Admit: false}
}

// ContainerRequest is the request of a container
type ContainerRequest struct {
	// PodMeta is the metadata of a pod
	PodMeta PodSandboxMetadata
	// ContainerMeta is the metadata of a container
	ContainerMeta ContainerMetadata
	// PodSandboxConfig is the config of a pod
	PodSandboxConfig utils.PodSandboxConfig
	// ContainerConfig is the config of a container
	ContainerConfig utils.ContainerConfig
	// ExtendedResourceConfiguration is the extended resource configuration of a container
	ExtendedResourceConfiguration *ExtendedResourceItem
	// CPUMilliRequest is the CPU request of a container
	CPUMilliRequest int64
}
type ContainerResponse struct {
	// Resources is the resources of a container
	Resources utils.LinuxContainerResources
	// AddContainerEnvs is the env of a container
	AddContainerEnvs map[string]string
	// AddContainerAnnotation is the annotation of a container
	AddContainerAnnotation map[string]string
}

// ConvertNRIRawPod2ContainerInfo  转化nriPod to containerInfo
func (c *ContainerInfo) ConvertNRIRawPod2ContainerInfo(pod *api.PodSandbox, container *api.Container) {
	c.ContainerRequest.PodMeta = PodSandboxMetadata{
		Name:      pod.GetName(),
		UID:       pod.GetUid(),
		Namespace: pod.GetNamespace(),
	}
	c.ContainerRequest.ContainerMeta = ContainerMetadata{
		Name: container.GetName(),
		ID:   container.GetId(),
	}
	// 计算CPU请求
	c.ContainerRequest.CPUMilliRequest = c.calculateCPUMilliRequest(container)

	GetContainerEngine(pod.GetAnnotations())
	ContainerCgroupParent, err := FixContainerCgroupParent(container)
	if err != nil {
		klog.Errorf("ConvertNRIRawPod2ContainerInfo  FixContainerCgroupParent error %v", err)
	}

	c.ContainerRequest.PodSandboxConfig = utils.PodSandboxConfig{
		Annotations: pod.GetAnnotations(),
		Labels:      pod.GetLabels(),
		Linux: &utils.LinuxPodSandboxConfig{
			CgroupParent: ContainerCgroupParent,
		},
	}

	c.ContainerRequest.ContainerConfig = utils.ContainerConfig{
		Envs:        GetContainerEnv(container.GetEnv()),
		Labels:      container.GetLabels(),
		Annotations: container.GetAnnotations(),
	}
	cfg, err := GetExtendedResourceConfiguration(pod.GetAnnotations())
	if err != nil {
		klog.Errorf("GetExtendedResourceConfiguration error %v", err)
	}
	if cfg != nil {
		if containerCfg, ok := cfg.Containers[container.Name]; ok {
			c.ContainerRequest.ExtendedResourceConfiguration = &containerCfg
		}
	}
}

// calculateCPUMilliRequest 计算容器的CPU毫核请求量
func (c *ContainerInfo) calculateCPUMilliRequest(container *api.Container) int64 {
	// 检查Linux CPU资源是否存在
	if !c.hasLinuxCPUResources(container) {
		klog.Warningf("Container %s: CPU resources not found. Using default: %d",
			container.GetId(), DefaultCPURequestMilliCores)
		return DefaultCPURequestMilliCores
	}

	cpuResources := container.GetLinux().GetResources().GetCpu()

	if cpuMilliFromQuota := c.calculateFromQuotaPeriod(container.GetId(), cpuResources); cpuMilliFromQuota > 0 {
		return cpuMilliFromQuota
	}

	if cpuMilliFromShares := c.calculateFromShares(container.GetId(), cpuResources); cpuMilliFromShares > 0 {
		return cpuMilliFromShares
	}

	klog.Warningf("Container %s: No CPU quota, period or shares. Using default: %d",
		container.GetId(), DefaultCPURequestMilliCores)
	return DefaultCPURequestMilliCores
}

// hasLinuxCPUResources 检查容器是否有Linux CPU资源配置
func (c *ContainerInfo) hasLinuxCPUResources(container *api.Container) bool {
	return container.GetLinux() != nil &&
		container.GetLinux().GetResources() != nil &&
		container.GetLinux().GetResources().GetCpu() != nil
}

// calculateFromQuotaPeriod 从quota和period计算CPU毫核数
func (c *ContainerInfo) calculateFromQuotaPeriod(containerID string, cpuResources *api.LinuxCPU) int64 {
	// 检查CPU资源是否存在
	if cpuResources == nil {
		return 0
	}
	defer func() {
		if r := recover(); r != nil {
			klog.Errorf("Container %s: panic in calculateFromQuotaPeriod: %v", containerID, r)
		}
	}()
	cpuQuota := cpuResources.GetQuota()
	if cpuQuota == nil {
		return 0
	}
	cpuPeriod := cpuResources.GetPeriod()
	if cpuPeriod == nil {
		return 0
	}

	if cpuQuota.Value > 0 && cpuPeriod.Value > 0 {
		cpuMilliRequest := int64(float64(cpuQuota.Value) / float64(cpuPeriod.Value) * float64(DefaultCPURequestMilliCores))
		return cpuMilliRequest
	}

	return 0
}

// calculateFromShares 从shares计算CPU毫核数
func (c *ContainerInfo) calculateFromShares(containerID string, cpuResources *api.LinuxCPU) int64 {
	// 检查Linux CPU资源是否存在
	if cpuResources == nil {
		return 0
	}
	defer func() {
		if r := recover(); r != nil {
			klog.Errorf("Container %s: panic in calculateFromShares: %v", containerID, r)
		}
	}()
	cpuShares := cpuResources.GetShares()
	if cpuShares == nil {
		return 0
	}
	if cpuShares.Value > 0 {
		cpuMilliRequest := int64(float64(cpuShares.Value) / float64(CPUSharesPerCore) * float64(DefaultCPURequestMilliCores))
		return cpuMilliRequest
	}

	return 0
}
