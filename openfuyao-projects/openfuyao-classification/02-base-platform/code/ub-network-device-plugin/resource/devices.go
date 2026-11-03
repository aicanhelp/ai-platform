/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package resource

import (
	"fmt"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

// Devices is a map of device ID to Device
type Devices map[string]*urmasdk.Device

// GetPluginDevices returns the plugin Devices from all devices in the Devices
func (ds Devices) GetPluginDevices() []*pluginapi.Device {
	var res []*pluginapi.Device
	var unhealthyDevices []string
	for _, device := range ds {
		health := pluginapi.Healthy
		if device.Healthy != true {
			health = pluginapi.Unhealthy
			unhealthyDevices = append(unhealthyDevices, fmt.Sprintf("%s(HwResId:%d)", device.Name, device.HwResId))
		}
		res = append(res, &pluginapi.Device{ID: device.Name, Health: health})
	}
	if len(unhealthyDevices) > 0 {
		klog.Warningf("Unhealthy devices: %v", unhealthyDevices)
	}
	return res
}

func (ds Devices) Equal(other Devices) bool {
	return devicesEqual(ds, other)
}

// devicesEqual compares two Devices maps by device ID set and Health status.
// It returns true if both maps contain the same devices with identical Health states.
func devicesEqual(a, b Devices) bool {
	if len(a) != len(b) {
		return false
	}

	for id, devA := range a {
		devB, ok := b[id]
		if !ok {
			return false
		}

		if devA.Healthy != devB.Healthy {
			return false
		}
	}

	return true
}
