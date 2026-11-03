/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package sshutil provides shared SSH connection helpers.
package sshutil

import (
	"net"
	"time"

	"golang.org/x/crypto/ssh"

	"env-check/pkg/config"
)

const defaultSSHPort = "22"

// DefaultPort returns the host SSH port, or "22" when unset.
func DefaultPort(host config.Host) string {
	if host.Port == "" {
		return defaultSSHPort
	}
	return host.Port
}

// Address returns the dial address for the host.
func Address(host config.Host) string {
	return net.JoinHostPort(host.IP, DefaultPort(host))
}

// BuildClientConfig builds an SSH client config for password authentication.
func BuildClientConfig(host config.Host, timeout time.Duration) *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: host.UserName,
		Auth: []ssh.AuthMethod{
			ssh.Password(host.Password),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         timeout,
	}
}
