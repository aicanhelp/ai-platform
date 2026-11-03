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

package runnermanager

import (
	"sync"

	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/runnermanager/cgroup"
	"openfuyao.com/colocation-management/pkg/utils"
)

// Runner is the interface for managing resources.
type Runner interface {
	SetCpuSet(cgroupPath string, resource utils.LinuxContainerResources) error
	SetCpuSetMem(cgroupPath string, resource utils.LinuxContainerResources) error
	SetCpuShares(cgroupPath string, resource utils.LinuxContainerResources) error
	SetCpuQuota(cgroupPath string, resource utils.LinuxContainerResources) error
	SetMemoryLimit(cgroupPath string, resource utils.LinuxContainerResources) error
}

var runner Runner
var once sync.Once

// GetRunner returns the runner instance.
func GetRunner() Runner {
	if runner != nil {
		return runner
	}
	once.Do(func() {
		klog.Infof("get Runner")
		runner = cgroup.NewV1CgroupManager()
	})
	return runner
}
