/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/yubo/golib/blob/v0.0.1/util/net/interface.go
*/

package net

import (
	"net"
	"strings"
)

const (
	// CIDRFormatParts is the expected number of parts when splitting a CIDR address
	// A CIDR address format is "IP/Mask" which has 2 parts when split by "/"
	CIDRFormatParts = 2
)

// networkInterface implements the networkInterfacer interface for production code, just
// wrapping the underlying net library function calls.
type networkInterface struct{}

func (_ networkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return net.InterfaceByName(intfName)
}

func (_ networkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return intf.Addrs()
}

func (_ networkInterface) Interfaces() ([]net.Interface, error) {
	return net.Interfaces()
}

// GetAllInterfaceIP returns all ip address of the host, including ipv4 and ipv6.
func GetAllInterfaceIP() ([]string, error) {
	nw := networkInterface{}
	intfs, err := nw.Interfaces()
	if err != nil {
		return nil, err
	}
	var ips []string
	for _, intf := range intfs {
		addrs, err := nw.Addrs(&intf)
		if err != nil {
			return nil, err
		}
		for _, addr := range addrs {
			ips = append(ips, addr.String())
		}
	}
	return ips, nil
}

// GetInterfaceFromIp returns the first interface name of the given IP address.
func GetInterfaceFromIp(ip string) (string, error) {
	nw := networkInterface{}
	intfs, err := nw.Interfaces()
	if err != nil {
		return "", err
	}
	for _, intf := range intfs {
		addrs, err := nw.Addrs(&intf)
		if err != nil {
			return "", err
		}
		for _, addr := range addrs {
			addrIPs := strings.Split(addr.String(), "/")
			if len(addrIPs) != CIDRFormatParts {
				continue
			}
			addrIP := addrIPs[0]
			if addrIP == ip {
				return intf.Name, nil
			}
		}
	}
	return "", nil
}
