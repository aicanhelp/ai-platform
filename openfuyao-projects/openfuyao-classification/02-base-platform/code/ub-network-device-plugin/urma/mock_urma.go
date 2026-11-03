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
	"errors"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"github.com/agiledragon/gomonkey/v2"
)

type MockUrma struct {
	UbsGetSharedDevicePatches *gomonkey.Patches
	UbsAllocateDevicePatches  *gomonkey.Patches

	ResetFunc []func()
}

func (m *MockUrma) Reset() {
	for _, fn := range m.ResetFunc {
		fn()
	}
}

func (m *MockUrma) MockUbsGetSharedDevice(name string) {
	switch name {
	case "UbsGetSharedDevice err":
		m.UbsGetSharedDevicePatches = gomonkey.ApplyFunc(urmasdk.UbsGetSharedDevice, func() ([]urmasdk.Device, error) {
			return []urmasdk.Device{}, errors.New("UbsGetSharedDevice err")
		})
	case "empty":
		m.UbsGetSharedDevicePatches = gomonkey.ApplyFunc(urmasdk.UbsGetSharedDevice, func() ([]urmasdk.Device, error) {
			return []urmasdk.Device{}, nil
		})
	default:
		m.UbsGetSharedDevicePatches = gomonkey.ApplyFunc(urmasdk.UbsGetSharedDevice, func() ([]urmasdk.Device, error) {
			devs := []urmasdk.Device{
				{"shared-1", true, 1},
				{"shared-2", false, 1},
			}
			return devs, nil
		})
	}

	m.ResetFunc = append(m.ResetFunc, m.UbsGetSharedDevicePatches.Reset)
}

func (m *MockUrma) MockUbsAllocateDevice(name string) {
	switch name {
	case "UbsAllocateDevice err":
		m.UbsAllocateDevicePatches = gomonkey.ApplyFunc(urmasdk.UbsAllocateDevice, func(name string) (urmasdk.DeviceInfo, error) {
			return urmasdk.DeviceInfo{}, errors.New("UbsAllocateDevice err")
		})
	default:
		m.UbsAllocateDevicePatches = gomonkey.ApplyFunc(urmasdk.UbsAllocateDevice, func() (urmasdk.DeviceInfo, error) {
			devInfo := urmasdk.DeviceInfo{
				VfePaths:    []string{"/dev/uburma/urma1", "/dev/uburma/urma2"},
				BondingPath: "/dev/uburma/bonding_dev_1",
				BondingEid:  "0000:0000:0000:0ae0:0010:0000",
			}
			return devInfo, nil
		})
	}

	m.ResetFunc = append(m.ResetFunc, m.UbsAllocateDevicePatches.Reset)
}
