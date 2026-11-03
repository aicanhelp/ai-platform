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
	"net"
	"os"
	"regexp"
	"strings"

	"github.com/pkg/errors"
)

const (
	// Using Google's public DNS server (8.8.8.8:53) for UDP connection
	DefaultDNSServer = "8.8.8.8:53"
	// IPv4NetworkPrefixStart is the starting index for network prefix comparison
	IPv4NetworkPrefixStart = 0
	// IPv4NetworkPrefixLength is the number of octets used for network prefix comparison
	IPv4NetworkPrefixLength = 3
)

func ValidIP(ip string) bool {
	if ip == "" {
		return false
	}
	if net.ParseIP(ip) != nil {
		return true
	}
	return false
}

// GetExternalIP 获取主机IP
func GetExternalIP() (string, error) {
	probeAddr := os.Getenv("EXTERNAL_IP_PROBE_ADDRESS")
	if probeAddr == "" {
		probeAddr = DefaultDNSServer
	}
	conn, err := net.Dial("udp", probeAddr)
	if err != nil {
		return "127.0.0.1", err
	}
	defer conn.Close()
	localAddr := conn.LocalAddr()
	udpAddr, ok := localAddr.(*net.UDPAddr)
	if !ok {
		return "", errors.New("failed to convert local address to UDPAddr")
	}
	ip := strings.Split(udpAddr.String(), ":")[0]
	return ip, nil
}

// SameNetworkSegment 判断两个IP是否在同一网段
// todo 是一个很笨的方法，但是暂时只有这样子
func SameNetworkSegment(ip1, ip2 string) bool {
	prefix1 := strings.Split(ip1, ".")[IPv4NetworkPrefixStart:IPv4NetworkPrefixLength]
	prefix2 := strings.Split(ip2, ".")[IPv4NetworkPrefixStart:IPv4NetworkPrefixLength]
	if strings.Join(prefix1, "") == strings.Join(prefix2, "") {
		return true
	}
	return false
}

const dns1123LabelFmt string = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"
const dns1123SubdomainFmt string = dns1123LabelFmt + "(\\." + dns1123LabelFmt + ")*"
const dns1123SubdomainErrorMsg string = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"

// DNS1123SubdomainMaxLength is a subdomain's max length in DNS (RFC 1123)
const DNS1123SubdomainMaxLength int = 253

var dns1123SubdomainRegexp = regexp.MustCompile("^" + dns1123SubdomainFmt + "$")

// IsDNS1123Subdomain tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123).
func IsDNS1123Subdomain(value string) error {
	if len(value) > DNS1123SubdomainMaxLength {
		return errors.Errorf("must be no more than %d characters", DNS1123SubdomainMaxLength)
	}
	if !dns1123SubdomainRegexp.MatchString(value) {
		return errors.Errorf("%q is not valid domain, a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com')", value)
	}
	return nil
}
