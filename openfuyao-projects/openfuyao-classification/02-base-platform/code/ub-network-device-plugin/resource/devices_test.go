/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package resource

import (
	"testing"

	urmasdkud "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"github.com/stretchr/testify/assert"
)

func TestDevicesEqual(t *testing.T) {
	tests := []struct {
		name     string
		devices1 Devices
		devices2 Devices
		want     bool
	}{
		{
			name:     "相同设备",
			devices1: Devices{"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}},
			devices2: Devices{"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}},
			want:     true,
		},
		{
			name:     "不同健康状态",
			devices1: Devices{"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}},
			devices2: Devices{"device1": &urmasdkud.Device{Name: "device1", Healthy: false, HwResId: 1}},
			want:     false,
		},
		{
			name:     "不同数量",
			devices1: Devices{"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}},
			devices2: Devices{
				"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
				"device2": &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2},
			},
			want: false,
		},
		{
			name:     "空设备",
			devices1: Devices{},
			devices2: Devices{},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.devices1.Equal(tt.devices2)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDevicesEqualFunc(t *testing.T) {
	devices1 := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
		"device2": &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2},
	}
	devices2 := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
		"device2": &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2},
	}

	assert.True(t, devicesEqual(devices1, devices2))
}

func TestDevicesGetPluginDevicesEmpty(t *testing.T) {
	devices := Devices{}
	pluginDevices := devices.GetPluginDevices()

	assert.Len(t, pluginDevices, 0)
}

func TestDevicesGetPluginDevicesHealthy(t *testing.T) {
	devices := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
	}
	pluginDevices := devices.GetPluginDevices()

	assert.Len(t, pluginDevices, 1)
	assert.Equal(t, "device1", pluginDevices[0].ID)
	assert.Equal(t, "Healthy", pluginDevices[0].Health)
}

func TestDevicesGetPluginDevicesUnhealthy(t *testing.T) {
	devices := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: false, HwResId: 1},
	}
	pluginDevices := devices.GetPluginDevices()

	assert.Len(t, pluginDevices, 1)
	assert.Equal(t, "device1", pluginDevices[0].ID)
	assert.Equal(t, "Unhealthy", pluginDevices[0].Health)
}

func TestDevicesGetPluginDevicesMixed(t *testing.T) {
	const expectedDeviceCount = 3
	devices := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
		"device2": &urmasdkud.Device{Name: "device2", Healthy: false, HwResId: 2},
		"device3": &urmasdkud.Device{Name: "device3", Healthy: true, HwResId: 3},
	}
	pluginDevices := devices.GetPluginDevices()

	assert.Len(t, pluginDevices, expectedDeviceCount)
	deviceMap := make(map[string]string)
	for _, d := range pluginDevices {
		deviceMap[d.ID] = d.Health
	}
	assert.Equal(t, "Healthy", deviceMap["device1"])
	assert.Equal(t, "Unhealthy", deviceMap["device2"])
	assert.Equal(t, "Healthy", deviceMap["device3"])
}

func TestDevicesEqualDeviceNotInB(t *testing.T) {
	devices1 := Devices{
		"device1": &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1},
	}
	devices2 := Devices{
		"device2": &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2},
	}

	assert.False(t, devicesEqual(devices1, devices2))
	assert.False(t, devicesEqual(devices2, devices1))
}
