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

package framework

import (
	"reflect"
	"strings"
	"testing"
)

func TestNodeRunnerArgv(t *testing.T) {
	cases := []struct {
		name string
		cfg  *Config
		node string
		ip   string
		cmd  string
		args []string
		want []string
	}{
		{
			name: "ssh uses node ip by default",
			cfg:  &Config{NodeAccess: NodeAccessSSH, NodeSSHUser: "root"},
			node: "node-1", ip: "10.1.1.5",
			cmd: "ctr", args: []string{"-n", "k8s.io", "containers", "ls"},
			want: []string{"ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
				"-o", "BatchMode=yes", "root@10.1.1.5", "ctr", "-n", "k8s.io", "containers", "ls"},
		},
		{
			name: "ssh host override wins over node ip",
			cfg:  &Config{NodeAccess: NodeAccessSSH, NodeSSHUser: "ops", NodeSSHHost: "bastion"},
			node: "node-1", ip: "10.1.1.5",
			cmd: "cat", args: []string{"/proc/1/cgroup"},
			want: []string{"ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
				"-o", "BatchMode=yes", "ops@bastion", "cat", "/proc/1/cgroup"},
		},
		{
			name: "docker execs into the kind node",
			cfg:  &Config{NodeAccess: NodeAccessDocker},
			node: "flux-sandbox-control-plane", ip: "172.19.0.2",
			cmd: "ctr", args: []string{"containers", "ls", "-q"},
			want: []string{"docker", "exec", "flux-sandbox-control-plane", "ctr", "containers", "ls", "-q"},
		},
		{
			name: "local runs the command directly",
			cfg:  &Config{NodeAccess: NodeAccessLocal},
			node: "node-1", ip: "10.1.1.5",
			cmd: "test", args: []string{"-f", "/var/log/flux-sandbox/x.log"},
			want: []string{"test", "-f", "/var/log/flux-sandbox/x.log"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := NewNodeRunner(tc.cfg, tc.node, tc.ip)
			got := n.argv(tc.cmd, tc.args...)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("argv = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNodeRunnerLocalRun(t *testing.T) {
	n := NewNodeRunner(&Config{NodeAccess: NodeAccessLocal}, "node-1", "10.1.1.5")
	out, err := n.Run("echo", "hello-node")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.TrimSpace(out) != "hello-node" {
		t.Errorf("Run output = %q, want hello-node", out)
	}
}

func TestNodeRunnerLocalRunShellAndFiles(t *testing.T) {
	n := NewNodeRunner(&Config{NodeAccess: NodeAccessLocal}, "node-1", "10.1.1.5")

	out, err := n.RunShell("printf 'a\nb\n'")
	if err != nil {
		t.Fatalf("RunShell: %v", err)
	}
	if out != "a\nb\n" {
		t.Errorf("RunShell output = %q", out)
	}

	if !n.PathExists("/etc/hosts") {
		t.Error("PathExists(/etc/hosts) = false, want true")
	}
	if n.PathExists("/no/such/file/here") {
		t.Error("PathExists(/no/such/file/here) = true, want false")
	}
	if n.GlobExists("/etc/hos*") != true {
		t.Error("GlobExists(/etc/hos*) = false, want true")
	}
	if n.GlobExists("/no/such/prefix-*") {
		t.Error("GlobExists(/no/such/prefix-*) = true, want false")
	}
}
