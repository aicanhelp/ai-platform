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

package pkiutil

import (
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	certutil "k8s.io/client-go/util/cert"
)

func TestGetIndexedIP(t *testing.T) {
	tests := []struct {
		name      string
		subnet    *net.IPNet
		index     int
		expected  string
		expectErr bool
	}{
		{
			name: "valid IPv4 subnet with index 1",
			subnet: &net.IPNet{
				IP:   net.ParseIP("10.0.0.0"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
			index:    1,
			expected: "10.0.0.1",
		},
		{
			name: "valid IPv4 subnet with index 0",
			subnet: &net.IPNet{
				IP:   net.ParseIP("192.168.1.0"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
			index:    0,
			expected: "192.168.1.0",
		},
		{
			name: "valid IPv4 subnet with index 10",
			subnet: &net.IPNet{
				IP:   net.ParseIP("10.0.0.0"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
			index:    10,
			expected: "10.0.0.10",
		},
		{
			name: "IPv6 subnet with index 1",
			subnet: &net.IPNet{
				IP:   net.ParseIP("fd00::"),
				Mask: net.IPMask([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0}),
			},
			index:    1,
			expected: "fd00::1",
		},
		{
			name: "index exceeds subnet range",
			subnet: &net.IPNet{
				IP:   net.ParseIP("10.0.0.0"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
			index:     256,
			expectErr: true,
		},
		{
			name: "small subnet with index 1",
			subnet: &net.IPNet{
				IP:   net.ParseIP("10.0.0.0"),
				Mask: net.IPv4Mask(255, 255, 255, 252),
			},
			index:    1,
			expected: "10.0.0.1",
		},
		{
			name: "small subnet with index 2",
			subnet: &net.IPNet{
				IP:   net.ParseIP("10.0.0.0"),
				Mask: net.IPv4Mask(255, 255, 255, 252),
			},
			index:    2,
			expected: "10.0.0.2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := GetIndexedIP(tt.subnet, tt.index)
			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result.String())
			}
		})
	}
}

func TestAppendSANsToAltNames(t *testing.T) {
	tests := []struct {
		name     string
		SANs     []string
		certName string
		wantDNS  []string
		wantIPs  []string
		wantErr  bool
	}{
		{
			name:     "valid DNS names",
			SANs:     []string{"example.com", "test.example.com"},
			certName: "test-cert",
			wantDNS:  []string{"example.com", "test.example.com"},
			wantIPs:  nil,
			wantErr:  false,
		},
		{
			name:     "valid IP addresses",
			SANs:     []string{"192.168.1.1", "10.0.0.1"},
			certName: "test-cert",
			wantDNS:  nil,
			wantIPs:  []string{"192.168.1.1", "10.0.0.1"},
			wantErr:  false,
		},
		{
			name:     "mixed DNS and IP",
			SANs:     []string{"example.com", "192.168.1.1"},
			certName: "test-cert",
			wantDNS:  []string{"example.com"},
			wantIPs:  []string{"192.168.1.1"},
			wantErr:  false,
		},
		{
			name:     "wildcard DNS",
			SANs:     []string{"*.example.com"},
			certName: "test-cert",
			wantDNS:  []string{"*.example.com"},
			wantIPs:  nil,
			wantErr:  false,
		},
		{
			name:     "invalid SAN",
			SANs:     []string{"invalid..example.com"},
			certName: "test-cert",
			wantErr:  true,
		},
		{
			name:     "empty SANs",
			SANs:     []string{},
			certName: "test-cert",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			altNames := &certutil.AltNames{}
			err := AppendSANsToAltNames(altNames, tt.SANs, tt.certName)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.wantDNS != nil {
					assert.Equal(t, tt.wantDNS, altNames.DNSNames)
				}
				if tt.wantIPs != nil {
					ipStrings := make([]string, len(altNames.IPs))
					for i, ip := range altNames.IPs {
						ipStrings[i] = ip.String()
					}
					assert.Equal(t, tt.wantIPs, ipStrings)
				}
			}
		})
	}
}

func TestAppendSANsToAltNamesWithNilSANs(t *testing.T) {
	altNames := &certutil.AltNames{
		DNSNames: []string{"existing.com"},
		IPs:      []net.IP{net.ParseIP("1.1.1.1")},
	}
	err := AppendSANsToAltNames(altNames, nil, "test-cert")
	assert.NoError(t, err)
	assert.Equal(t, []string{"existing.com"}, altNames.DNSNames)
}

func TestAppendSANsToAltNamesDeduplication(t *testing.T) {
	altNames := &certutil.AltNames{}
	SANs := []string{"example.com", "example.com", "192.168.1.1", "192.168.1.1"}
	err := AppendSANsToAltNames(altNames, SANs, "test-cert")
	assert.NoError(t, err)
	assert.Len(t, altNames.DNSNames, 1)
	assert.Len(t, altNames.IPs, 1)
}
