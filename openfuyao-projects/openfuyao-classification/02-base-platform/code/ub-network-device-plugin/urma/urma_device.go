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

package urma

import (
	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

func GetSharedDevice() ([]urmasdk.Device, error) {
	devices, err := urmasdk.UbsGetSharedDevice()
	if err != nil {
		return nil, err
	}

	return devices, nil
}

func AllocateDevice(name string) (urmasdk.DeviceInfo, error) {
	return urmasdk.UbsAllocateDevice(name)
}

func convertDevices(originalDevices []urmasdk.Device) []pluginapi.Device {
	convertedDevices := make([]pluginapi.Device, len(originalDevices))

	for i, originalDevice := range originalDevices {
		healthyStr := pluginapi.Healthy
		if !originalDevice.Healthy {
			healthyStr = pluginapi.Unhealthy
		}
		convertedDevice := pluginapi.Device{
			ID:     originalDevice.Name,
			Health: healthyStr,
		}
		convertedDevices[i] = convertedDevice
	}

	return convertedDevices
}
