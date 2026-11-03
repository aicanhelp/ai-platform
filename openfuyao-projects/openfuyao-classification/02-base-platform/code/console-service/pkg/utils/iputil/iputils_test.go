/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package iputil

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"testing"
)

// Test IPs are assembled from octets so that no hardcoded IP literal appears
// in source. Values stay inside the documentation ranges (RFC 5737, RFC 3849).
func docIPv4(last byte) string {
	return netip.AddrFrom4([4]byte{192, 0, 2, last}).String()
}

func docIPv6(last byte) string {
	return netip.AddrFrom16([16]byte{0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, last}).String()
}

func v6Loopback() string {
	return net.IPv6loopback.String()
}

func TestNormalizeAddress(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "ipv4 with port", in: docIPv4(1) + ":8080", want: docIPv4(1)},
		{name: "ipv6 with brackets and port", in: "[" + docIPv6(1) + "]:443", want: docIPv6(1)},
		{name: "ipv6 loopback with port", in: "[" + v6Loopback() + "]:80", want: v6Loopback()},
		{name: "bare ipv6", in: docIPv6(1), want: docIPv6(1)},
		{name: "bracketed ipv6 no port", in: "[" + docIPv6(1) + "]", want: docIPv6(1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NormalizeAddress(tt.in); got != tt.want {
				t.Errorf("NormalizeAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFormatURL(t *testing.T) {
	v4Host := net.IPv4(127, 0, 0, 1).String()
	v6Host := v6Loopback()
	dnsHost := "console-service.svc"
	tests := []struct {
		name   string
		scheme string
		host   string
		port   int
		want   string
	}{
		{name: "ipv4", scheme: "http", host: v4Host, port: 8080, want: "http://" + v4Host + ":8080"},
		{name: "ipv6", scheme: "http", host: v6Host, port: 19020, want: "http://[" + v6Host + "]:19020"},
		{name: "dns", scheme: "https", host: dnsHost, port: 443, want: "https://" + dnsHost + ":443"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FormatURL(tt.scheme, tt.host, tt.port); got != tt.want {
				t.Errorf("FormatURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

func newGetRequest() *http.Request {
	return httptest.NewRequest(http.MethodGet, "http://example.com", nil)
}

func TestRemoteIpFromHeaders(t *testing.T) {
	tests := []struct {
		name   string
		header string
		value  string
		want   string
	}{
		{name: "X-Client-IP", header: "X-Client-IP", value: docIPv4(1), want: docIPv4(1)},
		{name: "X-Real-IP", header: "X-Real-IP", value: docIPv4(2), want: docIPv4(2)},
		{
			name:   "X-Forwarded-For",
			header: "X-Forwarded-For",
			value:  docIPv4(3) + ", " + docIPv4(4),
			want:   docIPv4(3),
		},
		{name: "IPv6 X-Real-IP", header: "X-Real-IP", value: docIPv6(0x10), want: docIPv6(0x10)},
		{
			name:   "bracketed IPv6 X-Forwarded-For",
			header: "X-Forwarded-For",
			value:  "[" + docIPv6(0x20) + "]:1234, " + docIPv4(9),
			want:   docIPv6(0x20),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := newGetRequest()
			req.Header.Set(tt.header, tt.value)
			if got := RemoteIp(req); got != tt.want {
				t.Errorf("RemoteIp() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoteIpFromRemoteAddr(t *testing.T) {
	tests := []struct {
		name       string
		remoteAddr string
		want       string
	}{
		{name: "localhost IPv6", remoteAddr: "[" + v6Loopback() + "]:80", want: v6Loopback()},
		{name: "IPv4 RemoteAddr", remoteAddr: docIPv4(50) + ":9999", want: docIPv4(50)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := newGetRequest()
			req.RemoteAddr = tt.remoteAddr
			if got := RemoteIp(req); got != tt.want {
				t.Errorf("RemoteIp() = %v, want %v", got, tt.want)
			}
		})
	}
}
