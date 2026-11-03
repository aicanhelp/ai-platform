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
	"testing"

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
)

func TestWithResources(t *testing.T) {
	resources := &resource.Resources{}
	opt := WithResources(resources)

	o := &options{}
	opt(o)

	assert.Equal(t, resources, o.resources)
}

func TestOptions(t *testing.T) {
	resources := &resource.Resources{}
	o := &options{
		resources: resources,
	}

	assert.Equal(t, resources, o.resources)
}

func TestWithResourcesNil(t *testing.T) {
	opt := WithResources(nil)
	o := &options{}
	opt(o)
	assert.Nil(t, o.resources)
}

func TestMultipleOptions(t *testing.T) {
	resources1 := &resource.Resources{}
	resources2 := &resource.Resources{}
	o := &options{}
	WithResources(resources1)(o)
	WithResources(resources2)(o)
	assert.Equal(t, resources2, o.resources)
}

func TestOptionType(t *testing.T) {
	var _ Option = WithResources(nil)
}
