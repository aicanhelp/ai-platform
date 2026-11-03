/*
* Copyright (c) 2024 China Unicom Digital Technoklogy Co., Ltd.
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

// Package gpu
package gpu

import (
	"fmt"

	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
)

const (
	hookName = "gpu"
	// NvidiaDevices for NVIDIA devices
	NvidiaDevices = "NVIDIA_VISIBLE_DEVICES"
)

var configPath string

// GpuRunner is a GPU Runner
type GpuRunner struct {
}

var runner *GpuRunner

// NewGpuRunner creates a GPU Runner object
func NewGpuRunner() *GpuRunner {
	if runner != nil {
		return runner
	}
	runner = &GpuRunner{}
	return runner
}

// RegisterFunc register GpuRunner hooks
func (r *GpuRunner) RegisterFunc() {
	klog.Infof("register hook %v", hookName)
	_, err := hooks.RegisterFunc(configPath, common.PreCreateContainer, hookName, r.SetGpu)
	if err != nil {
		klog.Errorf("Error registering SetGpu: %v", err)
	}

}

// SetGpu sets the GPU info of GpuRunner
func (p *GpuRunner) SetGpu(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		return fmt.Errorf("info is not ContainerInfo,got type to %T", info)
	}
	annotations := containerInfo.ContainerRequest.PodSandboxConfig.Annotations
	// 添加从上报的数据中获取gpu可用的实例
	data, ok := annotations[common.GPUInstanceAnnotation]
	if !ok {
		return nil
	}

	if containerInfo.ContainerResponese.AddContainerEnvs == nil {
		containerInfo.ContainerResponese.AddContainerEnvs = make(map[string]string)
	}
	// 添加增加到annotation中
	containerInfo.ContainerResponese.AddContainerEnvs[NvidiaDevices] = data
	return nil
}
