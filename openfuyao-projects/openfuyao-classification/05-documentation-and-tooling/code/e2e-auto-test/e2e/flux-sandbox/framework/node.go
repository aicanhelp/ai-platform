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
	"bytes"
	"fmt"
	"log"
	"os/exec"
)

// NodeRunner executes commands on the k8s node hosting the agent pod. The
// transport is selected by Config.NodeAccess: ssh (standard Linux + k8s),
// docker (local kind), or local (the test process runs on the node).
type NodeRunner struct {
	mode       string
	sshTarget  string // user@host, ssh mode only
	dockerNode string // kind node (container) name, docker mode only
}

// NewNodeRunner builds a runner for the given node. nodeIP is the k8s node
// InternalIP used as the ssh target when no explicit host is configured.
func NewNodeRunner(cfg *Config, nodeName, nodeIP string) *NodeRunner {
	n := &NodeRunner{mode: cfg.NodeAccess, dockerNode: nodeName}
	if cfg.NodeAccess == NodeAccessSSH {
		host := cfg.NodeSSHHost
		if host == "" {
			host = nodeIP
		}
		n.sshTarget = cfg.NodeSSHUser + "@" + host
	}
	return n
}

// Run executes cmd with args on the node and returns stdout.
func (n *NodeRunner) Run(cmd string, args ...string) (string, error) {
	argv := n.argv(cmd, args...)
	c := exec.Command(argv[0], argv[1:]...)
	var stdout, stderr bytes.Buffer
	c.Stdout = &stdout
	c.Stderr = &stderr
	if err := c.Run(); err != nil {
		return stdout.String(), fmt.Errorf("%s: %w (stderr: %s)", cmd, err, stderr.String())
	}
	return stdout.String(), nil
}

// RunShell executes a shell script on the node and returns stdout.
func (n *NodeRunner) RunShell(script string) (string, error) {
	return n.Run("/bin/sh", "-c", script)
}

// BestEffort runs a command ignoring failures (used for force cleanup paths
// where the target may not exist). Errors are logged for debuggability.
func (n *NodeRunner) BestEffort(cmd string, args ...string) {
	if _, err := n.Run(cmd, args...); err != nil {
		log.Printf("best-effort %s %v (ignored): %v", cmd, args, err)
	}
}

// Ctr runs a ctr command against the k8s.io containerd namespace.
func (n *NodeRunner) Ctr(args ...string) (string, error) {
	return n.Run("ctr", append([]string{"-n", "k8s.io"}, args...)...)
}

// CtrBestEffort runs a ctr command ignoring failures.
func (n *NodeRunner) CtrBestEffort(args ...string) {
	n.BestEffort("ctr", append([]string{"-n", "k8s.io"}, args...)...)
}

// PathExists reports whether a file or directory exists on the node.
func (n *NodeRunner) PathExists(path string) bool {
	_, err := n.Run("test", "-e", path)
	return err == nil
}

// GlobExists reports whether any file matching the glob pattern exists on the
// node.
func (n *NodeRunner) GlobExists(pattern string) bool {
	out, err := n.RunShell("ls " + pattern + " 2>/dev/null")
	return err == nil && out != ""
}

// argv builds the full command line for Run.
func (n *NodeRunner) argv(cmd string, args ...string) []string {
	switch n.mode {
	case NodeAccessDocker:
		return append([]string{"docker", "exec", n.dockerNode, cmd}, args...)
	case NodeAccessLocal:
		return append([]string{cmd}, args...)
	default:
		argv := append([]string{"ssh"}, n.sshOptions()...)
		argv = append(argv, n.sshTarget, cmd)
		return append(argv, args...)
	}
}

func (n *NodeRunner) sshOptions() []string {
	return []string{
		"-o", "StrictHostKeyChecking=no",
		"-o", "UserKnownHostsFile=/dev/null",
		"-o", "BatchMode=yes",
	}
}
