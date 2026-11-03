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

// Package cgroup
package cgroup

import (
	"path/filepath"
	"strconv"
	"strings"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/runnermanager/cgroup/helper"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

var (
	cgroupFsAdapted = map[string]string{
		common.CPUSharesPath:   common.CgroupCPUDir,
		common.CPUQuotaPath:    common.CgroupCPUDir,
		common.CPUSetPath:      common.CgroupCPUSetDir,
		common.CPUSetMemPath:   common.CgroupCPUSetMemDir,
		common.MemoryLimitPath: common.CgroupMemDir,
	}
)

const (
	// DecimalBase is the decimal base.
	DecimalBase = 10
)

type v1CgroupManager struct {
	Config Config
}

// Config is the configuration for the cgroup manager.
type Config struct {
}

// NewV1CgroupManager creates a new v1cgroup manager.
func NewV1CgroupManager() *v1CgroupManager {
	return &v1CgroupManager{}
}

// SetCpuSet sets the cpu set for the specified cgroup path.
func (cm *v1CgroupManager) SetCpuSet(cgroupPath string, resource utils.LinuxContainerResources) error {
	// 添加
	// checkExist
	// valueCheck
	if resource.CPUSet == nil {
		return nil
	}
	if *resource.CPUSet == "" {
		return nil
	}
	// cgroupRead
	read, err := helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSetPath))
	if err != nil {
		return err
	}
	if !helper.IsCpuSetEqual(string(read), *resource.CPUSet) {
		// cgroupWrite
		err = helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
			common.CPUSetPath), *resource.CPUSet)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetCpuSetMem sets the cpu set mem for the specified cgroup path.
func (cm *v1CgroupManager) SetCpuSetMem(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CPUSetMem == nil {
		return nil
	}
	if *resource.CPUSetMem == "" {
		return nil
	}
	// cgroupRead
	read, err := helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSetMemPath))
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(read)) != strings.TrimSpace(*resource.CPUSetMem) {
		// cgroupWrite
		err = helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
			common.CPUSetMemPath), *resource.CPUSetMem)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetCpuShares sets the cpu shares for the specified cgroup path.
func (cm *v1CgroupManager) SetCpuShares(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CPUShares == nil {
		return nil
	}
	// 校验
	// cgroupRead
	read, err := helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSharesPath))
	if err != nil {
		return err
	}
	if string(read) == strconv.FormatInt(*resource.CPUShares, DecimalBase) {
		return nil
	}
	err = helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.CPUSharesPath), strconv.FormatInt(*resource.CPUShares, DecimalBase))
	if err != nil {
		return err
	}
	return nil
}

// SetCpuQuota sets the cpu quota for the specified cgroup path.
func (cm *v1CgroupManager) SetCpuQuota(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CFSQuota == nil {
		return nil
	}

	// cgroupRead
	read, err := helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUQuotaPath))
	if err != nil {
		return err
	}

	if string(read) == strconv.FormatInt(*resource.CFSQuota, DecimalBase) {
		return nil
	}
	err = helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.CPUQuotaPath), strconv.FormatInt(*resource.CFSQuota, DecimalBase))
	if err != nil {
		return err
	}
	return nil
}

// SetMemoryLimit sets the memory limit for the specified cgroup path.
func (cm *v1CgroupManager) SetMemoryLimit(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.MemoryLimit == nil {
		return nil
	}
	// cgroupRead
	read, err := helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.MemoryLimitPath))
	if err != nil {
		return err
	}

	if string(read) == strconv.FormatInt(*resource.MemoryLimit, DecimalBase) {
		return nil
	}
	err = helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.MemoryLimitPath), strconv.FormatInt(*resource.MemoryLimit, DecimalBase))
	if err != nil {
		return err
	}

	return nil
}

// CgroupConfig holds the cgroup configuration information.
// This is common object which is used to specify
// cgroup information to both systemd and raw cgroup fs
// implementation of the Cgroup Manager interface.
type CgroupConfig struct {
	// Fully qualified name prior to any driver specific conversions.
	Name v1CgroupManager
	// ResourceParameters contains various cgroups settings to apply.
	ResourceParameters *utils.LinuxContainerResources
}

// CgroupAbsPath returns the absolute path to the cgroup file.
func CgroupAbsPath(rootPath string, relativePath string, cgroupPath string) string {
	return filepath.Join(GetCgroupFilePath(rootPath, cgroupPath), relativePath, cgroupPath)
}

// GetCgroupFilePath
// kubepods.slice/kubepods-podxxxxxx.slice/
// cri-containerd-container1.scope
// Get /sys/fs/cgroup/cpu/kubepods.slice/kubepods-podxxxxxx.slice/
func GetCgroupFilePath(parent, cgroupPath string) string {
	// 添加 cgroupRootDir config
	return filepath.Join(common.DefaultCgroupRoot, cgroupFsAdapted[cgroupPath], parent)

}
