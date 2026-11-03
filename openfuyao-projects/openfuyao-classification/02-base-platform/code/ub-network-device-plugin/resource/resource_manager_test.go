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

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
)

func TestResourceManager_Resource(t *testing.T) {
	r := &resourceManager{
		name: Name(utils.ResourceName),
	}

	assert.Equal(t, Name(utils.ResourceName), r.Resource())
}

func TestResourceManager_Devices(t *testing.T) {
	devices := make(Devices)
	r := &resourceManager{
		devices: devices,
	}

	result := r.Devices()
	assert.Equal(t, devices, result)
}

func TestManagerInterface(t *testing.T) {
	var _ Manager = (*UrmaResourceManager)(nil)
}
