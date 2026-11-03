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

package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const testNodeIP = "192.168.1.1"

func TestXPUDeviceDefaultValues(t *testing.T) {
	dev := XPUDevice{}
	assert.Equal(t, int32(0), dev.Index)
	assert.Equal(t, "", dev.Id)
	assert.Equal(t, "", dev.Type)
	assert.Equal(t, false, dev.Health)
	assert.Equal(t, uint32(0), dev.Count)
	assert.Equal(t, "", dev.Mode)
	assert.Equal(t, int32(0), dev.Policy)
	assert.Equal(t, uint64(0), dev.MemoryTotal)
	assert.Equal(t, uint64(0), dev.MemoryUsed)
	assert.Equal(t, float64(0), dev.MemoryUtilization)
	assert.Equal(t, float64(0), dev.XpuUtilization)
	assert.Equal(t, "", dev.NodeName)
	assert.Equal(t, "", dev.NodeIp)
	assert.Equal(t, "", dev.DriverVersion)
	assert.Equal(t, 0, dev.FrameworkVersion)
	assert.Equal(t, 0, dev.PowerUsage)
	assert.Equal(t, 0, dev.Temperature)
	assert.Nil(t, dev.VxpuDeviceList)
}

func TestXPUDeviceSetAllFields(t *testing.T) {
	dev := XPUDevice{
		Index:             1,
		Id:                "xpu-0",
		Type:              "npu",
		Health:            true,
		Count:             2,
		Mode:              "exclusive",
		Policy:            1,
		MemoryTotal:       32768,
		MemoryUsed:        16384,
		MemoryUtilization: 50.0,
		XpuUtilization:    75.5,
		NodeName:          "node-1",
		NodeIp:            testNodeIP,
		DriverVersion:     "1.0.0",
		FrameworkVersion:  1,
		PowerUsage:        200,
		Temperature:       65,
		VxpuDeviceList: VxpuDevices{
			{Id: "vxpu-0", XpuId: "xpu-0", PodUID: "pod-1", ContainerName: "ctr-1",
				VxpuMemoryUsed: 8192, VxpuMemoryUtilization: 25.0, VxpuCoreUtilization: 30.0,
				VxpuMemoryLimit: 16384, VxpuCoreLimit: 50},
		},
	}

	assert.Equal(t, int32(1), dev.Index)
	assert.Equal(t, "xpu-0", dev.Id)
	assert.Equal(t, "npu", dev.Type)
	assert.True(t, dev.Health)
	assert.Equal(t, uint32(2), dev.Count)
	assert.Equal(t, "exclusive", dev.Mode)
	assert.Equal(t, int32(1), dev.Policy)
	assert.Equal(t, uint64(32768), dev.MemoryTotal)
	assert.Equal(t, uint64(16384), dev.MemoryUsed)
	assert.Equal(t, 50.0, dev.MemoryUtilization)
	assert.Equal(t, 75.5, dev.XpuUtilization)
	assert.Equal(t, "node-1", dev.NodeName)
	assert.Equal(t, testNodeIP, dev.NodeIp)
	assert.Equal(t, "1.0.0", dev.DriverVersion)
	assert.Equal(t, 1, dev.FrameworkVersion)
	assert.Equal(t, 200, dev.PowerUsage)
	assert.Equal(t, 65, dev.Temperature)
	assert.Len(t, dev.VxpuDeviceList, 1)
}

func TestVxpuDeviceDefaultValues(t *testing.T) {
	d := VxpuDevice{}
	assert.Equal(t, "", d.Id)
	assert.Equal(t, "", d.XpuId)
	assert.Equal(t, "", d.PodUID)
	assert.Equal(t, "", d.ContainerName)
	assert.Equal(t, uint64(0), d.VxpuMemoryUsed)
	assert.Equal(t, float64(0), d.VxpuMemoryUtilization)
	assert.Equal(t, float64(0), d.VxpuCoreUtilization)
	assert.Equal(t, int64(0), d.VxpuMemoryLimit)
	assert.Equal(t, int64(0), d.VxpuCoreLimit)
}

func TestVxpuDeviceSetFields(t *testing.T) {
	d := VxpuDevice{
		Id:                    "vxpu-1",
		XpuId:                 "xpu-0",
		PodUID:                "pod-abc",
		ContainerName:         "main",
		VxpuMemoryUsed:        4096,
		VxpuMemoryUtilization: 12.5,
		VxpuCoreUtilization:   20.0,
		VxpuMemoryLimit:       8192,
		VxpuCoreLimit:         25,
	}

	assert.Equal(t, "vxpu-1", d.Id)
	assert.Equal(t, "xpu-0", d.XpuId)
	assert.Equal(t, "pod-abc", d.PodUID)
	assert.Equal(t, "main", d.ContainerName)
	assert.Equal(t, uint64(4096), d.VxpuMemoryUsed)
	assert.Equal(t, 12.5, d.VxpuMemoryUtilization)
	assert.Equal(t, 20.0, d.VxpuCoreUtilization)
	assert.Equal(t, int64(8192), d.VxpuMemoryLimit)
	assert.Equal(t, int64(25), d.VxpuCoreLimit)
}

func TestVxpuDevicesSlice(t *testing.T) {
	devices := VxpuDevices{
		{Id: "vxpu-1", XpuId: "xpu-0"},
		{Id: "vxpu-2", XpuId: "xpu-1"},
	}

	assert.Len(t, devices, 2)
	assert.Equal(t, "vxpu-1", devices[0].Id)
	assert.Equal(t, "vxpu-2", devices[1].Id)
}

func TestVxpuDevicesEmpty(t *testing.T) {
	devices := VxpuDevices{}
	assert.Len(t, devices, 0)
}
