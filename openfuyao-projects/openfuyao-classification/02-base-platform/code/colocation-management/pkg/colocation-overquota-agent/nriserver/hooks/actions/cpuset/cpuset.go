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

// Package cpuset
package cpuset

import (
	"fmt"

	"k8s.io/klog/v2"
	"k8s.io/utils/pointer"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
)

// CpuSetRunner is the runner of cpuset.
type CpuSetRunner struct {
}

const (
	hookName = "cpuset"
)

var configPath string

var runner *CpuSetRunner

// NewCpuSetRunner is the constructor of CpuSetRunner.
func NewCpuSetRunner() *CpuSetRunner {
	if runner != nil {
		return runner
	}
	runner = &CpuSetRunner{}
	return runner
}

// RegisterFunc is the function to register the hook.
func (r *CpuSetRunner) RegisterFunc() {
	klog.Infof("register hook %v", hookName)
	var err error
	_, err = hooks.RegisterFunc(configPath, common.PreRunPodSandbox, hookName, r.UnsetPodCPUQuota)
	if err != nil {
		klog.Errorf("Error registering UnsetPodCPUQuota: %v", err)
	}
	_, err = hooks.RegisterFunc(configPath, common.PreCreateContainer, hookName, r.SetContainerCPUSet)
	if err != nil {
		klog.Errorf("Error registering SetContainerCPUSet: %v", err)
	}
	_, err = hooks.RegisterFunc(configPath, common.PreUpdateContainerResources, hookName, r.SetContainerCPUSet)
	if err != nil {
		klog.Errorf("Error registering SetContainerCPUSet: %v", err)
	}
}

// SetContainerCPUSet is the function to set container cpu set.
func (p *CpuSetRunner) SetContainerCPUSet(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		return fmt.Errorf("info is not ContainerInfo,got type to %T", info)
	}
	set, err := getContainerCPUSet(&containerInfo.ContainerRequest)
	if err != nil {
		return err
	}
	if *set == "" {
		return nil
	}
	containerInfo.ContainerResponese.Resources.CPUSet = set
	return nil

}

// UnsetPodCPUQuota is the function to unset pod cpu quota.
func (p *CpuSetRunner) UnsetPodCPUQuota(info core.Info) error {
	podInfo, ok := info.(*core.PodInfo)
	if !ok {
		return fmt.Errorf("info is not PodInfo,got type to %T", info)
	}
	cpuset, err := core.GetPodCPUset(podInfo.PodRequest.PodSandboxConfig.Annotations)
	if err != nil {
		return err
	}
	if cpuset == "" {
		return nil
	}
	podInfo.PodResponese.Resources.CFSQuota = pointer.Int64(-1)
	return nil
}

func getContainerCPUSet(containerReq *core.ContainerRequest) (*string, error) {
	// 获取qos Level
	level := core.GetPodQosLevel(containerReq.PodSandboxConfig.Annotations)

	if level == common.QosBe {
		return pointer.String(""), nil
	} else {
		// 添加注解里获取NUMANodeResources
		return pointer.String(""), nil
	}
}
