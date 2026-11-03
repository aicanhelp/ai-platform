/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package normalize

import (
	"fmt"
	"net"
	"strconv"
	"strings"
)

const (
	minimumPort = 1
	maximumPort = 65535
)

// Architecture maps architecture aliases to the two canonical values
// "amd64" or "arm64" (design spec §0.11.3). Unknown inputs return "other" so
// the evaluator can fail a platform check rather than silently mismatch.
func Architecture(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "amd64", "x86_64", "x64":
		return "amd64"
	case "arm64", "aarch64":
		return "arm64"
	default:
		return "other"
	}
}

// OSFamily maps OS family aliases to canonical lowercase values.
func OSFamily(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "linux", "gnu/linux":
		return "linux"
	case "windows":
		return "windows"
	case "darwin", "macos":
		return "darwin"
	default:
		return "other"
	}
}

// Distribution maps distribution aliases to canonical lowercase IDs
// (design spec §0.11.3). UOS variants fold to "uos"; openEuler/Ubuntu fold to
// lowercase.
func Distribution(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "openeuler":
		return "openeuler"
	case "ubuntu":
		return "ubuntu"
	case "uos", "uniontech os":
		return "uos"
	default:
		// Preserve an unknown lowercase value rather than guessing; the
		// evaluator's `in` check will simply not match the supported set.
		return strings.ToLower(strings.TrimSpace(s))
	}
}

// Address removes equivalence differences between addresses (design
// spec §0.5). It lowercases hostnames, strips IPv6 brackets to a canonical
// form, and normalizes a bare IPv4/IPv6 so that equivalent representations
// (e.g. "192.0.2.1" vs "192.0.2.001") hash identically. Non-IP strings are
// lowercased and trimmed only.
func Address(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// Strip a scheme/host suffix is the caller's responsibility; here we only
	// canonicalize the host part.
	host := s
	// Lowercase hostnames (IPv4 is case-insensitive already; IPv6 is not, but
	// Go's net.ParseIP is case-insensitive for hex digits in practice).
	host = strings.ToLower(host)
	if ip := net.ParseIP(strings.Trim(host, "[]")); ip != nil {
		return ip.String()
	}
	return host
}

// ParsePort parses a port string into an int in [1,65535]. It rejects
// non-numeric or out-of-range values, which become INPUT_ERROR run issues
// rather than silently defaulting.
func ParsePort(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty port")
	}
	p, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("port %q is not a number", s)
	}
	if p < minimumPort || p > maximumPort {
		return 0, fmt.Errorf("port %d out of range 1..65535", p)
	}
	return p, nil
}
