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
	"testing"

	urmasdkud "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
)

func TestDeviceMapInsert(t *testing.T) {
	deviceMap := make(DeviceMap)
	name := Name("unifiedbus.com/test")
	device := &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}

	deviceMap.insert(name, device)

	assert.NotNil(t, deviceMap[name])
	assert.Equal(t, device, deviceMap[name]["device1"])
}

func TestDeviceMapInsertMultiple(t *testing.T) {
	deviceMap := make(DeviceMap)
	name := Name("unifiedbus.com/test")
	device1 := &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}
	device2 := &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2}

	deviceMap.insert(name, device1)
	deviceMap.insert(name, device2)

	assert.Len(t, deviceMap[name], 2)
	assert.Equal(t, device1, deviceMap[name]["device1"])
	assert.Equal(t, device2, deviceMap[name]["device2"])
}

func TestDeviceMapInsertDifferentNames(t *testing.T) {
	deviceMap := make(DeviceMap)
	name1 := Name("unifiedbus.com/test1")
	name2 := Name("unifiedbus.com/test2")
	device1 := &urmasdkud.Device{Name: "device1", Healthy: true, HwResId: 1}
	device2 := &urmasdkud.Device{Name: "device2", Healthy: true, HwResId: 2}

	deviceMap.insert(name1, device1)
	deviceMap.insert(name2, device2)

	assert.Len(t, deviceMap, 2)
	assert.Len(t, deviceMap[name1], 1)
	assert.Len(t, deviceMap[name2], 1)
}

func TestNewDeviceMapSuccess(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	deviceMap, err := NewDeviceMap(resources)
	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
	assert.Len(t, deviceMap, 1)
	assert.Len(t, deviceMap["unifiedbus.com/test"], 2)
}

func TestNewDeviceMapGetSharedDeviceError(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("UbsGetSharedDevice err")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	deviceMap, err := NewDeviceMap(resources)
	assert.Error(t, err)
	assert.Nil(t, deviceMap)
	assert.Contains(t, err.Error(), "error getting shared devices")
}

func TestNewDeviceMapPatternNotMatch(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "nonexistent-*", Name: "unifiedbus.com/test"},
		},
	}

	deviceMap, err := NewDeviceMap(resources)
	assert.Error(t, err)
	assert.Nil(t, deviceMap)
	assert.Contains(t, err.Error(), "does not match any resource patterns")
}

func TestNewDeviceMapEmptyResources(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{},
	}

	deviceMap, err := NewDeviceMap(resources)
	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
	assert.Len(t, deviceMap, 0)
}

func TestNewDeviceMapEmptySharedDevices(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("empty")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	deviceMap, err := NewDeviceMap(resources)
	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
	assert.Len(t, deviceMap, 0)
}

func TestDeviceMapBuilderBuild(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	b := &deviceMapBuilder{resources: resources}
	deviceMap, err := b.build()

	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
}

func TestDeviceMapBuilderBuildDeviceMapFromDefaultResources(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	b := &deviceMapBuilder{resources: resources}
	deviceMap, err := b.buildDeviceMapFromDefaultResources()

	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
}

func TestDeviceMapBuilderBuildSharedDeviceMapSuccess(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("success")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	b := &deviceMapBuilder{resources: resources}
	deviceMap, err := b.buildSharedDeviceMap()

	assert.NoError(t, err)
	assert.NotNil(t, deviceMap)
	assert.Len(t, deviceMap["unifiedbus.com/test"], 2)
}

func TestDeviceMapBuilderBuildSharedDeviceMapError(t *testing.T) {
	mock := urma.MockUrma{}
	mock.MockUbsGetSharedDevice("UbsGetSharedDevice err")
	defer mock.Reset()

	resources := &Resources{
		SHAREDs: []Resource{
			{Pattern: "*", Name: "unifiedbus.com/test"},
		},
	}

	b := &deviceMapBuilder{resources: resources}
	deviceMap, err := b.buildSharedDeviceMap()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error getting shared devices")
	assert.Nil(t, deviceMap)
}
