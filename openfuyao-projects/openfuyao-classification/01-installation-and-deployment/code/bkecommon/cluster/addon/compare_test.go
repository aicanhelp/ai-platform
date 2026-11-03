/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package addon

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	tAddons = Addons{
		{
			Name:    "kubeproxy",
			Version: "1.21.1",
			Block:   false,
		},
		{
			Name:    "abc",
			Version: "1.21.1",
		},
	}
)

func TestCompareBKEConfigAddon(t *testing.T) {
	tests := getCompareBKEConfigAddonTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCompareBKEConfigAddonTest(t, tt)
		})
	}
}

type addonTestCase struct {
	name    string
	args    Addons
	operate AddonOperate
	expect  int
}

func getCompareBKEConfigAddonTestCases() []addonTestCase {
	return []addonTestCase{
		getCreateAddonTestCase(),
		getUpdateAddonTestCase(),
		getRemoveAddonTestCase(),
		getUpgradeAddonTestCase(),
	}
}

func getCreateAddonTestCase() addonTestCase {
	return addonTestCase{
		name: "create",
		args: Addons{
			{
				Name:    "abc",
				Version: "1.21.1",
			},
			{
				Name: "ass",
			},
			{
				Name:    "kubeproxy",
				Version: "1.21.2",
				Block:   true,
			},
		},
		operate: CreateAddon,
		expect:  1,
	}
}

func getUpdateAddonTestCase() addonTestCase {
	return addonTestCase{
		name: "update",
		args: Addons{
			{
				Name:    "kubeproxy",
				Version: "1.21.2",
				Block:   false,
			},
			{
				Name:    "abc",
				Version: "1.21.1",
				Block:   true,
			},
		},
		operate: UpdateAddon,
		expect:  0,
	}
}

func getRemoveAddonTestCase() addonTestCase {
	return addonTestCase{
		name:    "remove",
		args:    Addons{},
		operate: RemoveAddon,
		expect:  2,
	}
}

func getUpgradeAddonTestCase() addonTestCase {
	return addonTestCase{
		name: "upgrade",
		args: Addons{
			{
				Name:    "kubeproxy",
				Version: "1.21.2",
			},
		},
		operate: UpgradeAddon,
		expect:  1,
	}
}

func runCompareBKEConfigAddonTest(t *testing.T, tt addonTestCase) {
	if got, ok := CompareBKEConfigAddon(tAddons, tt.args); ok {
		var d int
		for _, g := range got {
			if g.Operate != tt.operate {
				continue
			}
			d += 1
		}
		assert.Equal(t, d, tt.expect, "expect get %d addons, but get %d", tt.expect, d)
	} else {
		t.Errorf("expected addon change but got nil")
	}
}
