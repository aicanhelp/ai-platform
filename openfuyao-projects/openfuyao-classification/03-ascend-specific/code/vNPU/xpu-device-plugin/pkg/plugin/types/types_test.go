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

package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDeviceBindTimeConstant(t *testing.T) {
	assert.Equal(t, "huawei.com/bind-time", DeviceBindTime)
}

func TestDeviceBindPhaseConstant(t *testing.T) {
	assert.Equal(t, "huawei.com/bind-phase", DeviceBindPhase)
}

func TestDeviceBindStatusConstants(t *testing.T) {
	assert.Equal(t, "allocating", DeviceBindAllocating)
	assert.Equal(t, "failed", DeviceBindFailed)
	assert.Equal(t, "success", DeviceBindSuccess)
}

func TestVXPULockNameConstant(t *testing.T) {
	assert.Equal(t, "vnpu/mutex.lock", VXPULockName)
}

func TestContainerDeviceZeroValue(t *testing.T) {
	cd := ContainerDevice{}
	assert.Equal(t, int32(0), cd.Index)
	assert.Equal(t, "", cd.UUID)
	assert.Equal(t, "", cd.Type)
	assert.Equal(t, int32(0), cd.Usedmem)
	assert.Equal(t, int32(0), cd.Usedcores)
	assert.Equal(t, int32(0), cd.Vid)
	assert.Equal(t, "", cd.Template)
	assert.Equal(t, int32(0), cd.Policy)
}

func TestContainerDeviceFieldAssignment(t *testing.T) {
	cd := ContainerDevice{
		Index:     1,
		UUID:      "uuid-001",
		Type:      "npu",
		Usedmem:   1024,
		Usedcores: 4,
		Vid:       2,
		Template:  "tpl",
		Policy:    1,
	}
	assert.Equal(t, int32(1), cd.Index)
	assert.Equal(t, "uuid-001", cd.UUID)
	assert.Equal(t, "npu", cd.Type)
	assert.Equal(t, int32(1024), cd.Usedmem)
	assert.Equal(t, int32(4), cd.Usedcores)
	assert.Equal(t, int32(2), cd.Vid)
	assert.Equal(t, "tpl", cd.Template)
	assert.Equal(t, int32(1), cd.Policy)
}

func TestContainerDevicesSliceBehavior(t *testing.T) {
	var cds = ContainerDevices{}
	assert.Equal(t, 0, len(cds))

	cds = append(cds, ContainerDevice{Index: 0, UUID: "u0"})
	cds = append(cds, ContainerDevice{Index: 1, UUID: "u1"})
	assert.Equal(t, 2, len(cds))
	assert.Equal(t, "u0", cds[0].UUID)
	assert.Equal(t, "u1", cds[1].UUID)
}

func TestPodDevicesSliceBehavior(t *testing.T) {
	var pods = PodDevices{}
	assert.Equal(t, 0, len(pods))

	pods = append(pods, ContainerDevices{
		ContainerDevice{Index: 0, UUID: "u0"},
		ContainerDevice{Index: 1, UUID: "u1"},
	})
	pods = append(pods, ContainerDevices{
		{Index: 2, UUID: "u2"},
	})
	assert.Equal(t, 2, len(pods))
	assert.Equal(t, 2, len(pods[0]))
	assert.Equal(t, 1, len(pods[1]))
	assert.Equal(t, "u2", pods[1][0].UUID)
}

func TestDeviceInfoZeroValue(t *testing.T) {
	di := DeviceInfo{}
	assert.Equal(t, int32(0), di.Index)
	assert.Equal(t, "", di.Id)
	assert.Equal(t, int32(0), di.Count)
	assert.Equal(t, int32(0), di.Devmem)
	assert.Equal(t, "", di.Type)
	assert.Equal(t, false, di.Health)
	assert.Equal(t, int32(0), di.Numa)
	assert.Equal(t, "", di.Mode)
	assert.Equal(t, int32(0), di.Corenum)
}

func TestDeviceInfoFieldAssignment(t *testing.T) {
	di := DeviceInfo{
		Index:   3,
		Id:      "dev-003",
		Count:   8,
		Devmem:  65536,
		Type:    "npu",
		Health:  true,
		Numa:    1,
		Mode:    "default",
		Corenum: 32,
	}
	assert.Equal(t, int32(3), di.Index)
	assert.Equal(t, "dev-003", di.Id)
	assert.Equal(t, int32(8), di.Count)
	assert.Equal(t, int32(65536), di.Devmem)
	assert.Equal(t, "npu", di.Type)
	assert.True(t, di.Health)
	assert.Equal(t, int32(1), di.Numa)
	assert.Equal(t, "default", di.Mode)
	assert.Equal(t, int32(32), di.Corenum)
}

func TestXPUDeviceZeroValue(t *testing.T) {
	xd := XPUDevice{}
	assert.Equal(t, int32(0), xd.Index)
	assert.Equal(t, "", xd.Id)
	assert.Equal(t, "", xd.Type)
	assert.False(t, xd.Health)
	assert.Equal(t, uint32(0), xd.Count)
	assert.Equal(t, "", xd.Mode)
	assert.Equal(t, int32(0), xd.Policy)
	assert.Equal(t, uint64(0), xd.MemoryTotal)
	assert.Equal(t, uint64(0), xd.MemoryUsed)
	assert.Equal(t, float64(0), xd.MemoryUtilization)
	assert.Equal(t, float64(0), xd.XpuUtilization)
	assert.Equal(t, "", xd.NodeName)
	assert.Equal(t, "", xd.NodeIp)
	assert.Equal(t, "", xd.DriverVersion)
	assert.Equal(t, 0, xd.FrameworkVersion)
	assert.Equal(t, uint32(0), xd.PowerUsage)
	assert.Equal(t, uint32(0), xd.Temperature)
	assert.Equal(t, 0, len(xd.VxpuDeviceList))
}

func TestXPUDeviceFieldAssignment(t *testing.T) {
	vList := VxpuDevices{
		{Id: "vxpu-0"},
	}
	xd := XPUDevice{
		Index:             0,
		Id:                "xpu-0",
		Type:              "npu",
		Health:            true,
		Count:             1,
		Mode:              "default",
		Policy:            0,
		MemoryTotal:       8589934592,
		MemoryUsed:        1073741824,
		MemoryUtilization: 12.5,
		XpuUtilization:    33.3,
		NodeName:          "node-a",
		NodeIp:            "127.0.0.1",
		DriverVersion:     "1.0.0",
		FrameworkVersion:  2,
		PowerUsage:        80,
		Temperature:       45,
		VxpuDeviceList:    vList,
	}
	assert.Equal(t, int32(0), xd.Index)
	assert.Equal(t, "xpu-0", xd.Id)
	assert.Equal(t, "npu", xd.Type)
	assert.True(t, xd.Health)
	assert.Equal(t, uint32(1), xd.Count)
	assert.Equal(t, "default", xd.Mode)
	assert.Equal(t, int32(0), xd.Policy)
	assert.Equal(t, uint64(8589934592), xd.MemoryTotal)
	assert.Equal(t, uint64(1073741824), xd.MemoryUsed)
	assert.InDelta(t, 12.5, xd.MemoryUtilization, 0.0001)
	assert.InDelta(t, 33.3, xd.XpuUtilization, 0.0001)
	assert.Equal(t, "node-a", xd.NodeName)
	assert.Equal(t, "127.0.0.1", xd.NodeIp)
	assert.Equal(t, "1.0.0", xd.DriverVersion)
	assert.Equal(t, 2, xd.FrameworkVersion)
	assert.Equal(t, uint32(80), xd.PowerUsage)
	assert.Equal(t, uint32(45), xd.Temperature)
	assert.Equal(t, 1, len(xd.VxpuDeviceList))
	assert.Equal(t, "vxpu-0", xd.VxpuDeviceList[0].Id)
}

func TestVxpuDeviceZeroValue(t *testing.T) {
	vd := VxpuDevice{}
	assert.Equal(t, "", vd.Id)
	assert.Equal(t, "", vd.XpuId)
	assert.Equal(t, "", vd.PodUID)
	assert.Equal(t, "", vd.ContainerName)
	assert.Equal(t, uint64(0), vd.VxpuMemoryUsed)
	assert.Equal(t, float64(0), vd.VxpuMemoryUtilization)
	assert.Equal(t, float64(0), vd.VxpuCoreUtilization)
	assert.Equal(t, int64(0), vd.VxpuMemoryLimit)
	assert.Equal(t, int64(0), vd.VxpuCoreLimit)
	assert.Equal(t, int32(0), vd.Policy)
}

func TestVxpuDeviceFieldAssignment(t *testing.T) {
	vd := VxpuDevice{
		Id:                    "v-0",
		XpuId:                 "xpu-0",
		PodUID:                "pod-uid-0",
		ContainerName:         "c0",
		VxpuMemoryUsed:        512,
		VxpuMemoryUtilization: 6.25,
		VxpuCoreUtilization:   12.5,
		VxpuMemoryLimit:       8192,
		VxpuCoreLimit:         4,
		Policy:                1,
	}
	assert.Equal(t, "v-0", vd.Id)
	assert.Equal(t, "xpu-0", vd.XpuId)
	assert.Equal(t, "pod-uid-0", vd.PodUID)
	assert.Equal(t, "c0", vd.ContainerName)
	assert.Equal(t, uint64(512), vd.VxpuMemoryUsed)
	assert.InDelta(t, 6.25, vd.VxpuMemoryUtilization, 0.0001)
	assert.InDelta(t, 12.5, vd.VxpuCoreUtilization, 0.0001)
	assert.Equal(t, int64(8192), vd.VxpuMemoryLimit)
	assert.Equal(t, int64(4), vd.VxpuCoreLimit)
	assert.Equal(t, int32(1), vd.Policy)
}

func TestVxpuDevicesSliceBehavior(t *testing.T) {
	var list = VxpuDevices{}
	assert.Equal(t, 0, len(list))

	list = append(list, VxpuDevice{Id: "a"})
	list = append(list, VxpuDevice{Id: "b"})
	assert.Equal(t, 2, len(list))
	assert.Equal(t, "a", list[0].Id)
	assert.Equal(t, "b", list[1].Id)
}

func TestProcessUsageZeroValue(t *testing.T) {
	pu := ProcessUsage{}
	assert.Equal(t, uint64(0), pu.ProcessMem)
	assert.Equal(t, uint64(0), pu.ProcessCoreUtilization)
}

func TestProcessUsageFieldAssignment(t *testing.T) {
	pu := ProcessUsage{
		ProcessMem:             1024,
		ProcessCoreUtilization: 50,
	}
	assert.Equal(t, uint64(1024), pu.ProcessMem)
	assert.Equal(t, uint64(50), pu.ProcessCoreUtilization)
}

func TestDeviceUsageInfoZeroValue(t *testing.T) {
	du := DeviceUsageInfo{}
	assert.Equal(t, uint32(0), du.CoreUtil)
	assert.Equal(t, uint32(0), du.MemUtil)
	assert.Equal(t, uint32(0), du.PowerUsage)
	assert.Equal(t, uint32(0), du.Temperature)
}

func TestDeviceUsageInfoFieldAssignment(t *testing.T) {
	du := DeviceUsageInfo{
		CoreUtil:    60,
		MemUtil:     40,
		PowerUsage:  100,
		Temperature: 50,
	}
	assert.Equal(t, uint32(60), du.CoreUtil)
	assert.Equal(t, uint32(40), du.MemUtil)
	assert.Equal(t, uint32(100), du.PowerUsage)
	assert.Equal(t, uint32(50), du.Temperature)
}

func TestNodeConfigZeroValue(t *testing.T) {
	nc := NodeConfig{}
	assert.Equal(t, "", nc.Name)
	assert.Equal(t, "", nc.Mode)
}

func TestNodeConfigFieldAssignment(t *testing.T) {
	nc := NodeConfig{
		Name: "node-a",
		Mode: "default",
	}
	assert.Equal(t, "node-a", nc.Name)
	assert.Equal(t, "default", nc.Mode)
}

func TestNodeConfigDataZeroValue(t *testing.T) {
	ncd := NodeConfigData{}
	assert.Equal(t, 0, len(ncd.NodeConfigs))
}

func TestNodeConfigDataFieldAssignment(t *testing.T) {
	ncd := NodeConfigData{
		NodeConfigs: []NodeConfig{
			NodeConfig{Name: "node-a", Mode: "default"},
			NodeConfig{Name: "node-b", Mode: "boost"},
		},
	}
	assert.Equal(t, 2, len(ncd.NodeConfigs))
	assert.Equal(t, "node-a", ncd.NodeConfigs[0].Name)
	assert.Equal(t, "default", ncd.NodeConfigs[0].Mode)
	assert.Equal(t, "node-b", ncd.NodeConfigs[1].Name)
	assert.Equal(t, "boost", ncd.NodeConfigs[1].Mode)
}
