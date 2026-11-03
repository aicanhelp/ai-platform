/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package net

import (
	"net"
	"syscall"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

const (
	numZero = 0
	numOne  = 1
	numTwo  = 2
	numFour = 4
)

func TestRemoveRepIP(t *testing.T) {
	tests := []struct {
		name     string
		ips      []net.IP
		expected int
	}{
		{
			name:     "empty slice",
			ips:      []net.IP{},
			expected: 0,
		},
		{
			name:     "single IP",
			ips:      []net.IP{net.ParseIP("192.168.1.1")},
			expected: 1,
		},
		{
			name:     "duplicate IPs",
			ips:      []net.IP{net.ParseIP("192.168.1.1"), net.ParseIP("192.168.1.1")},
			expected: 1,
		},
		{
			name:     "multiple unique IPs",
			ips:      []net.IP{net.ParseIP("192.168.1.1"), net.ParseIP("192.168.1.2"), net.ParseIP("192.168.1.3")},
			expected: 3,
		},
		{
			name:     "mixed duplicates and unique",
			ips:      []net.IP{net.ParseIP("192.168.1.1"), net.ParseIP("192.168.1.2"), net.ParseIP("192.168.1.1"), net.ParseIP("192.168.1.3")},
			expected: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveRepIP(tt.ips)
			assert.Equal(t, tt.expected, len(result))
		})
	}
}

func TestRemoveRepIPWithNil(t *testing.T) {
	result := RemoveRepIP(nil)
	assert.Equal(t, 0, len(result))
}

func TestRemoveRepDomain(t *testing.T) {
	tests := []struct {
		name     string
		domains  []string
		expected int
	}{
		{
			name:     "empty slice",
			domains:  []string{},
			expected: 0,
		},
		{
			name:     "single domain",
			domains:  []string{"example.com"},
			expected: 1,
		},
		{
			name:     "duplicate domains",
			domains:  []string{"example.com", "example.com"},
			expected: 1,
		},
		{
			name:     "multiple unique domains",
			domains:  []string{"example.com", "test.com", "foo.com"},
			expected: 3,
		},
		{
			name:     "mixed duplicates and unique",
			domains:  []string{"example.com", "test.com", "example.com", "foo.com"},
			expected: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveRepDomain(tt.domains)
			assert.Equal(t, tt.expected, len(result))
		})
	}
}

func TestRemoveRepDomainWithNil(t *testing.T) {
	result := RemoveRepDomain(nil)
	assert.Equal(t, 0, len(result))
}

func TestInterfaceIpExitWithExistingIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockInterface := &net.Interface{
		Name:  "eth0",
		Index: 1,
	}

	mockAddrs := []net.Addr{
		&net.IPNet{
			IP:   net.ParseIP("192.168.1.1"),
			Mask: net.IPv4Mask(255, 255, 255, 0),
		},
	}

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		if name == "eth0" {
			return mockInterface, nil
		}
		return nil, &net.OpError{Op: "route", Net: "ip", Source: nil, Addr: nil, Err: syscall.ENOENT}
	})

	patches.ApplyMethodFunc(mockInterface, "Addrs", func() ([]net.Addr, error) {
		return mockAddrs, nil
	})

	result, err := InterfaceIpExit("eth0", "192.168.1.1")
	assert.NoError(t, err)
	assert.Contains(t, result, "192.168.1.1")
}

func TestInterfaceIpExitWithNonExistingIP(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockInterface := &net.Interface{
		Name:  "eth0",
		Index: 1,
	}

	mockAddrs := []net.Addr{
		&net.IPNet{
			IP:   net.ParseIP("192.168.1.1"),
			Mask: net.IPv4Mask(255, 255, 255, 0),
		},
	}

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		if name == "eth0" {
			return mockInterface, nil
		}
		return nil, &net.OpError{Op: "route", Net: "ip", Source: nil, Addr: nil, Err: syscall.ENOENT}
	})

	patches.ApplyMethodFunc(mockInterface, "Addrs", func() ([]net.Addr, error) {
		return mockAddrs, nil
	})

	result, err := InterfaceIpExit("eth0", "10.0.0.1")
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestInterfaceIpExitWithError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		return nil, &net.OpError{Op: "route", Net: "ip", Source: nil, Addr: nil, Err: syscall.ENOENT}
	})

	result, err := InterfaceIpExit("nonexistent", "192.168.1.1")
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestInterfaceIpExitWithAddrsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockInterface := &net.Interface{
		Name:  "eth0",
		Index: 1,
	}

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		return mockInterface, nil
	})

	patches.ApplyMethodFunc(mockInterface, "Addrs", func() ([]net.Addr, error) {
		return nil, &net.OpError{Op: "route", Net: "ip", Source: nil, Addr: nil, Err: syscall.ENOENT}
	})

	result, err := InterfaceIpExit("eth0", "192.168.1.1")
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestInterfaceIpExitEmptyInterfaceName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		return nil, &net.OpError{Op: "route", Net: "ip", Source: nil, Addr: nil, Err: syscall.ENOENT}
	})

	result, err := InterfaceIpExit("", "192.168.1.1")
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestInterfaceIpExitMultipleAddresses(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockInterface := &net.Interface{
		Name:  "eth0",
		Index: 1,
	}

	mockAddrs := []net.Addr{
		&net.IPNet{
			IP:   net.ParseIP("192.168.1.1"),
			Mask: net.IPv4Mask(255, 255, 255, 0),
		},
		&net.IPNet{
			IP:   net.ParseIP("192.168.1.2"),
			Mask: net.IPv4Mask(255, 255, 255, 0),
		},
	}

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		return mockInterface, nil
	})

	patches.ApplyMethodFunc(mockInterface, "Addrs", func() ([]net.Addr, error) {
		return mockAddrs, nil
	})

	result, err := InterfaceIpExit("eth0", "192.168.1.1")
	assert.NoError(t, err)
	assert.Equal(t, "192.168.1.1/24", result)
}

func TestInterfaceIpExit_IPPrefixBoundary(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockInterface := &net.Interface{
		Name:  "eth0",
		Index: 1,
	}

	mockAddrs := []net.Addr{
		&net.IPNet{
			IP:   net.ParseIP("122.235.189.16"),
			Mask: net.IPv4Mask(255, 255, 255, 0),
		},
		&net.IPNet{
			IP:   net.ParseIP("122.235.189.1"),
			Mask: net.IPv4Mask(255, 255, 255, 255),
		},
	}

	patches.ApplyFunc(net.InterfaceByName, func(name string) (*net.Interface, error) {
		return mockInterface, nil
	})
	patches.ApplyMethodFunc(mockInterface, "Addrs", func() ([]net.Addr, error) {
		return mockAddrs, nil
	})

	result, err := InterfaceIpExit("eth0", "122.235.189.1")
	assert.NoError(t, err)
	assert.Equal(t, "122.235.189.1/32", result)
}
