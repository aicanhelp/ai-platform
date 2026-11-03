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

package local

import (
	"net"
	"os/exec"
	"strconv"
)

// canBind reports whether a TCP or UDP port is free on the loopback address.
// It attempts a bind and closes immediately; this is read-only (no listener
// is kept open). For UDP, a successful bind means the port is free.
func canBind(protocol string, port int) bool {
	switch protocol {
	case "udp":
		conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: port})
		if err != nil {
			return false
		}
		return conn.Close() == nil
	default: // tcp
		ln, err := net.Listen("tcp", joinHostPort("0.0.0.0", port))
		if err != nil {
			return false
		}
		return ln.Close() == nil
	}
}

// joinHostPort builds a host:port string.
func joinHostPort(host string, port int) string {
	return net.JoinHostPort(host, itoa(port))
}

func itoa(value int) string {
	return strconv.Itoa(value)
}

// lookPath wraps exec.LookPath so it can be replaced in tests.
func lookPath(name string) (string, error) {
	return exec.LookPath(name)
}
