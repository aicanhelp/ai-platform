/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
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

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

func TestAddressMatchesIP(t *testing.T) {
	assert.True(t, AddressMatchesIP("122.235.189.16/24", "122.235.189.16"))
	assert.False(t, AddressMatchesIP("122.235.189.16/24", "122.235.189.1"))
	assert.True(t, AddressMatchesIP("192.168.1.1", "192.168.1.1"))
	assert.False(t, AddressMatchesIP("192.168.1.1/24", "192.168.1.2"))
}

func TestResolveLocalNodeIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockIPs := []string{
		"10.244.1.5/32",
		"192.168.1.10/24",
		"127.0.0.1/8",
	}
	patches.ApplyFunc(GetAllInterfaceIP, func() ([]string, error) {
		return mockIPs, nil
	})

	t.Run("prefer candidate intersection over first interface IP", func(t *testing.T) {
		ip, err := ResolveLocalNodeIP("192.168.1.10")
		assert.NoError(t, err)
		assert.Equal(t, "192.168.1.10", ip)
	})

	t.Run("fallback to first usable IPv4 without candidates", func(t *testing.T) {
		ip, err := ResolveLocalNodeIP()
		assert.NoError(t, err)
		assert.Equal(t, "10.244.1.5", ip)
	})

	t.Run("no match when candidate not on host", func(t *testing.T) {
		ip, err := ResolveLocalNodeIP("203.0.113.1")
		assert.NoError(t, err)
		assert.Equal(t, "", ip)
	})
}

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
