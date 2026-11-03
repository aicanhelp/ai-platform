/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package core

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"openfuyao.com/colocation-management/pkg/common"
)

// ExtendedResourceConfiguration is a struct that represents the extended resource configuration.
type ExtendedResourceConfiguration struct {
	// 	container为key
	InitContainers map[string]ExtendedResourceItem `json:"initContainers,omitempty"`
	// container为key
	Containers map[string]ExtendedResourceItem `json:"containers,omitempty"`
}

// ExtendedResourceItem  is a struct that represents the extended resource item.
type ExtendedResourceItem struct {
	Requests v1.ResourceList `json:"requests,omitempty"`
	Limits   v1.ResourceList `json:"limits,omitempty"`
}

// SetupTest is a helper function to set up test cases.
func SetupTest(t *testing.T) (map[v1.ResourceName]resource.Quantity, ExtendedResourceConfiguration, []byte) {
	cpu := resource.MustParse("2000")
	memory := resource.MustParse("2Gi")

	containerConfiguration := map[string]ExtendedResourceItem{}
	request := map[v1.ResourceName]resource.Quantity{}
	request[common.ExtenderResourceCPU] = cpu
	request[common.ExtenderResourceMemory] = memory
	limit := map[v1.ResourceName]resource.Quantity{}
	limit[common.ExtenderResourceCPU] = cpu
	limit[common.ExtenderResourceMemory] = memory
	containerConfiguration["container1"] = ExtendedResourceItem{
		Requests: request,
		Limits:   limit,
	}
	configuration := ExtendedResourceConfiguration{
		Containers: containerConfiguration,
	}
	marshal, err := json.Marshal(configuration)
	assert.NoError(t, err)

	return request, configuration, marshal
}
