/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package cgroup

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func TestNewV1CgroupManagerNotNil(t *testing.T) {
	assert.NotNil(t, NewV1CgroupManager())
}

func TestGetCgroupFilePathSegments(t *testing.T) {
	p := GetCgroupFilePath("kubepods.slice", common.CPUSharesPath)
	assert.True(t, strings.HasSuffix(p, "kubepods.slice"))
	assert.Contains(t, GetCgroupFilePath("kubepods.slice", common.CPUSetPath), "cpuset")
}

func TestCgroupAbsPathContainsFile(t *testing.T) {
	assert.Contains(t, CgroupAbsPath("parent", "rel", common.CPUSharesPath), "cpu.shares")
}

func TestSetCpuSetMemGuards(t *testing.T) {
	cm := NewV1CgroupManager()
	empty := ""
	assert.NoError(t, cm.SetCpuSetMem("p", utils.LinuxContainerResources{}))
	assert.NoError(t, cm.SetCpuSetMem("p", utils.LinuxContainerResources{CPUSetMem: &empty}))
}

func TestSetCpuQuotaNilGuard(t *testing.T) {
	cm := NewV1CgroupManager()
	assert.NoError(t, cm.SetCpuQuota("p", utils.LinuxContainerResources{}))
}

func TestSetMemoryLimitNilGuard(t *testing.T) {
	cm := NewV1CgroupManager()
	assert.NoError(t, cm.SetMemoryLimit("p", utils.LinuxContainerResources{}))
}

func TestSetCpuSharesNilGuard(t *testing.T) {
	cm := NewV1CgroupManager()
	assert.NoError(t, cm.SetCpuShares("p", utils.LinuxContainerResources{}))
}

func TestSetCpuSetNilGuard(t *testing.T) {
	cm := NewV1CgroupManager()
	empty := ""
	assert.NoError(t, cm.SetCpuSet("p", utils.LinuxContainerResources{}))
	assert.NoError(t, cm.SetCpuSet("p", utils.LinuxContainerResources{CPUSet: &empty}))
}

// value paths: the constructed cgroup path does not exist under the host
// /sys/fs/cgroup root, so CgroupFileRead deterministically fails and each
// Set* returns that error (no host mutation occurs). Assert the error rather
// than discarding it, so the read path is genuinely exercised/verified.
func TestSetMethodsWithValues(t *testing.T) {
	cm := NewV1CgroupManager()
	cs := "0-3"
	mem := "0"
	shares := int64(1024)
	quota := int64(100000)
	limit := int64(1 << 30)
	assert.Error(t, cm.SetCpuSet("nonexistent-test-cgroup", utils.LinuxContainerResources{CPUSet: &cs}))
	assert.Error(t, cm.SetCpuSetMem("nonexistent-test-cgroup", utils.LinuxContainerResources{CPUSetMem: &mem}))
	assert.Error(t, cm.SetCpuShares("nonexistent-test-cgroup", utils.LinuxContainerResources{CPUShares: &shares}))
	assert.Error(t, cm.SetCpuQuota("nonexistent-test-cgroup", utils.LinuxContainerResources{CFSQuota: &quota}))
	assert.Error(t, cm.SetMemoryLimit("nonexistent-test-cgroup", utils.LinuxContainerResources{MemoryLimit: &limit}))
}
