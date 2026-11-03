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
package net

import (
	"testing"
)

func TestInterface(t *testing.T) {
	t.Run("GetAllInterfaceIP", func(t *testing.T) {
		intf, err := GetAllInterfaceIP()
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log(intf)
	})

	t.Run("GetInterfaceFromIp", func(t *testing.T) {
		ip := "127.0.0.1"
		intf, err := GetInterfaceFromIp(ip)
		if err != nil {
			t.Fatal(err)
			return
		}
		t.Log(intf)
	})

	t.Run("InterfaceByName", func(t *testing.T) {
		ni := new(networkInterface)
		ni.InterfaceByName("eth0")
	})
}
