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

	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
)

type deviceMapBuilder struct {
	resources *Resources
}

type DeviceMap map[Name]Devices

func NewDeviceMap(resources *Resources) (DeviceMap, error) {
	b := deviceMapBuilder{
		resources: resources,
	}

	return b.build()
}

// build builds a map of name names to devices.
func (b *deviceMapBuilder) build() (DeviceMap, error) {
	devices, err := b.buildDeviceMapFromDefaultResources()
	if err != nil {
		return nil, fmt.Errorf("error building device map from config.resources: %v", err)
	}

	return devices, nil
}

// buildDeviceMapFromDefaultResources builds a map of name names to devices from spec.Config.Resources
func (b *deviceMapBuilder) buildDeviceMapFromDefaultResources() (DeviceMap, error) {
	sharedDeviceMap, err := b.buildSharedDeviceMap()
	if err != nil {
		return nil, fmt.Errorf("error building Shared device map: %v", err)
	}

	return sharedDeviceMap, nil
}

// buildSharedDeviceMap builds a map of name names to shared devices
func (b *deviceMapBuilder) buildSharedDeviceMap() (DeviceMap, error) {
	deviceMap := make(DeviceMap)

	sharedDevices, err := urma.GetSharedDevice()
	if err != nil {
		return nil, fmt.Errorf("error getting shared devices: %v", err)
	}

	for _, sharedDevice := range sharedDevices {
		for _, resource := range b.resources.SHAREDs {
			name := sharedDevice.Name
			if resource.Pattern.Matches(name) {
				deviceMap.insert(resource.Name, &sharedDevice)
			} else {
				return deviceMap, fmt.Errorf("SHARED name '%v' does not match any resource patterns", name)
			}
		}
	}

	return deviceMap, nil
}

func (d DeviceMap) insert(name Name, dev *urmasdk.Device) {
	if d[name] == nil {
		d[name] = make(Devices)
	}
	d[name][dev.Name] = dev
}
