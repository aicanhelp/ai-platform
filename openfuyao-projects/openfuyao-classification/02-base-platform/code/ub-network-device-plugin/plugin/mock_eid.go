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

package plugin

import (
	"github.com/agiledragon/gomonkey/v2"
)

type MockEid struct {
	GetBondingEidPatches *gomonkey.Patches

	ResetFunc []func()
}

func (m *MockEid) Reset() {
	for _, fn := range m.ResetFunc {
		fn()
	}
}

func (m *MockEid) mockGetBondingEid() {
	m.GetBondingEidPatches = gomonkey.ApplyFunc((*urmaDevicePlugin).GetBondingEid, func(p *urmaDevicePlugin, deviceID string) (string, error) {
		return "eid", nil
	})

	m.ResetFunc = append(m.ResetFunc, m.GetBondingEidPatches.Reset)
}
