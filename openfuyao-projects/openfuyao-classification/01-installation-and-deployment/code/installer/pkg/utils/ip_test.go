/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"fmt"
	"net"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

// TestGetLocalIP tests the GetLocalIP function.
func TestGetLocalIPSuccess(t *testing.T) {
	patches := gomonkey.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		addrs := []net.Addr{
			&net.IPNet{IP: net.IPv4(192, 168, 1, 100), Mask: net.CIDRMask(24, 32)},
			&net.IPNet{IP: net.IPv4(127, 0, 0, 1), Mask: net.CIDRMask(8, 32)},
			&net.IPNet{IP: net.IPv6loopback, Mask: net.CIDRMask(128, 128)},
		}
		return addrs, nil
	})
	defer patches.Reset()

	ip, err := GetLocalIP()
	assert.NoError(t, err)
	assert.Equal(t, "192.168.1.100", ip)
}

// TestGetLocalIPNoValidIP tests the GetLocalIP function when there is no valid IP.
func TestGetLocalIPNoValidIP(t *testing.T) {
	patches := gomonkey.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		addrs := []net.Addr{
			&net.IPNet{IP: net.IPv4(127, 0, 0, 1), Mask: net.CIDRMask(8, 32)},
			&net.IPNet{IP: net.IPv6loopback, Mask: net.CIDRMask(128, 128)},
		}
		return addrs, nil
	})
	defer patches.Reset()

	ip, err := GetLocalIP()
	assert.Error(t, err)
	assert.Equal(t, "valid local IP not found!", err.Error())
	assert.Equal(t, "", ip)
}

// TestGetLocalIPError tests the GetLocalIP function when there is an error.
func TestGetLocalIPError(t *testing.T) {
	patches := gomonkey.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		return nil, fmt.Errorf("mock error")
	})
	defer patches.Reset()

	ip, err := GetLocalIP()
	assert.Error(t, err)
	assert.Equal(t, "mock error", err.Error())
	assert.Equal(t, "", ip)
}
