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
)

func TestAddon(t *testing.T) {
	t.Run("filter", func(t *testing.T) {
		addon := Addons{
			{
				Name:    "kubeproxy",
				Version: "1.21.2",
			},
		}
		fo := FilterOptions{}
		addon.Filter(fo)
	})

	t.Run("length", func(t *testing.T) {
		addon := Addons{
			{
				Name:    "kubeproxy",
				Version: "1.21.2",
			},
		}
		addon.Length()
	})
}
