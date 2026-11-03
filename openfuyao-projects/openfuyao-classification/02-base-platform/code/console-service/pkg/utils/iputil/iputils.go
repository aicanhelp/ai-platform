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
	"strconv"
	"strings"

	"console-service/pkg/zlog"
)

// NormalizeAddress strips an optional port and brackets from host / host:port / [ipv6]:port.
// IPv4 and IPv6 are preserved as-is (e.g. "::1" stays "::1").
func NormalizeAddress(address string) string {
	address = strings.TrimSpace(address)
	if address == "" {
		return address
	}
	if host, _, err := net.SplitHostPort(address); err == nil {
		return host
	}
	return strings.Trim(address, "[]")
}

// FormatURL builds scheme://host:port. IPv6 literals are bracketed via net.JoinHostPort.
func FormatURL(scheme, host string, port int) string {
	return scheme + "://" + net.JoinHostPort(host, strconv.Itoa(port))
}

// RemoteIp Tries to extract the client IP from provided headers (IPv4 and IPv6).
func RemoteIp(req *http.Request) string {
	address := req.RemoteAddr
	if ip := req.Header.Get("x-client-ip"); ip != "" {
		address = ip
	} else if ip := req.Header.Get("X-Real-IP"); ip != "" {
		address = ip
	} else if ip := req.Header.Get("X-Forwarded-For"); ip != "" {
		xffIps := strings.Split(ip, ",")
		if len(xffIps) > 0 {
			address = strings.TrimSpace(xffIps[0])
		}
	} else {
		host, _, err := net.SplitHostPort(address)
		if err != nil {
			zlog.Errorf("Unable to get remoteAddr: %v", err)
			return NormalizeAddress(address)
		}
		address = host
	}

	return NormalizeAddress(address)
}
