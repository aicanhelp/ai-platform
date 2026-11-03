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

// Package beresource
package beresource

import (
	"fmt"

	"k8s.io/klog/v2"
	"k8s.io/utils/pointer"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

const (
	hookName = "extendedResource"
)

var configPath string

// BEResourceRunner is the runner of BEResource.
type BEResourceRunner struct {
}

var runner *BEResourceRunner

// NewBEResourceRunner returns a new BEResourceRunner.
func NewBEResourceRunner() *BEResourceRunner {
	if runner != nil {
		return runner
	}
	runner = &BEResourceRunner{}
	return runner
}

// RegisterFunc registers the BEResourceRunner.
func (r *BEResourceRunner) RegisterFunc() {
	klog.Infof("register hook %v", hookName)
	var err error
	_, err = hooks.RegisterFunc(configPath, common.PreRunPodSandbox, hookName, r.SetPodResources)
	if err != nil {
		klog.Errorf("Error registering SetPodResources: %v", err)
	}
	_, err = hooks.RegisterFunc(configPath, common.PreCreateContainer, hookName, r.SetContainerResources)
	if err != nil {
		klog.Errorf("Error registering SetContainerResources: %v", err)
	}
	_, err = hooks.RegisterFunc(configPath, common.PreUpdateContainerResources, hookName, r.SetContainerResources)
	if err != nil {
		klog.Errorf("Error registering SetContainerResources: %v", err)
	}
}

// SetPodResources sets the pod resources.
func (r *BEResourceRunner) SetPodResources(info core.Info) error {
	var allErrs []error
	err1 := r.SetPodCPUShares(info)
	if err1 != nil {
		klog.Errorf("%v hook failed to set PodCpuShares ,err: %v", hookName, err1)
	}

	allErrs = append(allErrs, err1)
	err2 := r.SetPodCFSQuota(info)
	if err2 != nil {
		klog.Errorf("%v hook failed to SetPodCFSQuota ,err: %v", hookName, err2)

	}
	allErrs = append(allErrs, err2)
	err3 := r.SetPodMemoryLimit(info)
	if err3 != nil {
		klog.Errorf("%v hook failed to SetPodMemoryLimit ,err: %v", hookName, err3)
	}
	allErrs = append(allErrs, err3)
	return utils.AggregateErrors(allErrs)

}

// SetContainerResources sets the container resources.
func (r *BEResourceRunner) SetContainerResources(info core.Info) error {
	var allErrs []error
	err1 := r.SetContainerCPUShares(info)
	if err1 != nil {
		klog.Errorf("%v hook failed to set SetContainerCPUShares ,err: %v", hookName, err1)
	}
	allErrs = append(allErrs, err1)
	err2 := r.SetContainerCFSQuota(info)
	if err2 != nil {
		klog.Errorf("%v hook failed to SetContainerCFSQuota ,err: %v", hookName, err2)

	}
	allErrs = append(allErrs, err2)
	err3 := r.SetContainerMemoryLimit(info)
	if err3 != nil {
		klog.Errorf("%v hook failed to SetContainerMemoryLimit ,err: %v", hookName, err3)
	}
	allErrs = append(allErrs, err3)
	return utils.AggregateErrors(allErrs)
}

// SetPodCPUShares sets the pod cpu shares.
func (r *BEResourceRunner) SetPodCPUShares(info core.Info) error {
	podInfo, ok := info.(*core.PodInfo)
	if !ok {
		return fmt.Errorf("info is not PodInfo,got type to %T", info)
	}

	if core.GetPodQosLevel(podInfo.PodRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(podInfo.PodRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	request := int64(0)
	for _, container := range configuration.Containers {
		if container.Requests != nil {

			if fixcpu, ok := container.Requests[common.ExtenderResourceCPU]; ok {
				request += fixcpu.Value()
			} else {
				continue
			}
		}
		continue
	}
	shares := MilliCPUToShares(request)
	podInfo.PodResponese.Resources.CPUShares = pointer.Int64(shares)
	return nil
}

// SetPodCFSQuota sets the pod cfs quota.
func (r *BEResourceRunner) SetPodCFSQuota(info core.Info) error {
	podInfo, ok := info.(*core.PodInfo)
	if !ok {
		return fmt.Errorf("info is not PodInfo,got type to %T", info)
	}

	if core.GetPodQosLevel(podInfo.PodRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(podInfo.PodRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	limit := int64(0)
	for _, container := range configuration.Containers {
		if container.Limits != nil {

			if fixcpu, ok := container.Limits[common.ExtenderResourceCPU]; ok {
				limit += fixcpu.Value()
			} else {
				limit = -1
				break
			}
		} else {
			limit = -1
			break
		}
	}
	quota := MilliCPUToQuota(limit)

	podInfo.PodResponese.Resources.CFSQuota = pointer.Int64(quota)
	return nil

}

// SetPodMemoryLimit sets the pod memory limit.
func (r *BEResourceRunner) SetPodMemoryLimit(info core.Info) error {
	podInfo, ok := info.(*core.PodInfo)
	if !ok {
		return fmt.Errorf("info is not PodInfo,got type to %T", info)
	}
	if core.GetPodQosLevel(podInfo.PodRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(podInfo.PodRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	memory := int64(0)
	for _, container := range configuration.Containers {
		if container.Limits != nil {
			if fixmemory, ok := container.Limits[common.ExtenderResourceMemory]; ok {
				memory += fixmemory.Value()
			} else {
				memory = -1
				break
			}
		} else {
			memory = -1
			break
		}
	}
	podInfo.PodResponese.Resources.MemoryLimit = pointer.Int64(memory)
	return nil
}

// SetContainerCPUShares sets the container cpu shares.
func (r *BEResourceRunner) SetContainerCPUShares(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		return fmt.Errorf("info is not ContainerInfo,got type to %T", info)
	}
	if core.GetPodQosLevel(containerInfo.ContainerRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(
		containerInfo.ContainerRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	request := int64(0)
	requests := configuration.Containers[containerInfo.ContainerRequest.ContainerMeta.Name].Requests
	if requests != nil {
		if fixcpu, ok := requests[common.ExtenderResourceCPU]; ok {
			request += fixcpu.Value()
		}
	}
	shares := MilliCPUToShares(request)
	containerInfo.ContainerResponese.Resources.CPUShares = pointer.Int64(shares)
	return nil
}

// SetContainerCFSQuota sets the container cfs quota.
func (r *BEResourceRunner) SetContainerCFSQuota(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		return fmt.Errorf("info is not ContainerInfo,got type to %T", info)
	}
	if core.GetPodQosLevel(containerInfo.ContainerRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(
		containerInfo.ContainerRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	limit := int64(0)
	limits := configuration.Containers[containerInfo.ContainerRequest.ContainerMeta.Name].Limits
	if limits != nil {
		if fixlimit, ok := limits[common.ExtenderResourceCPU]; ok {
			limit += fixlimit.Value()
		}
	}
	quota := MilliCPUToQuota(limit)
	containerInfo.ContainerResponese.Resources.CFSQuota = pointer.Int64(quota)
	return nil
}

// SetContainerMemoryLimit sets the container memory limit.
func (r *BEResourceRunner) SetContainerMemoryLimit(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		return fmt.Errorf("info is not ContainerInfo,got type to %T", info)
	}

	if core.GetPodQosLevel(containerInfo.ContainerRequest.PodSandboxConfig.Annotations) != common.QosBe {
		return nil
	}

	configuration, err := core.GetExtendedResourceConfiguration(
		containerInfo.ContainerRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	memory := int64(0)
	memorys := configuration.Containers[containerInfo.ContainerRequest.ContainerMeta.Name].Limits
	if memorys != nil {
		if fixmemory, ok := memorys[common.ExtenderResourceMemory]; ok {
			memory += fixmemory.Value()
		}
	}
	containerInfo.ContainerResponese.Resources.MemoryLimit = pointer.Int64(memory)
	return nil
}
