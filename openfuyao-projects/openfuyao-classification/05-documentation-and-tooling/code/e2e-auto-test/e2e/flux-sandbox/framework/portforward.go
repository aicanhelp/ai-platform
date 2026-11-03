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
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

// PortForward manages a `kubectl port-forward` subprocess exposing a pod
// port on 127.0.0.1 (e.g. the agent HTTP API).
type PortForward struct {
	cmd        *exec.Cmd
	namespace  string
	podName    string
	localPort  int
	remotePort int
}

// StartPortForward starts kubectl port-forward mapping localPort to
// remotePort on the pod and waits until ready returns nil (or fails after
// 60s). ready is typically a healthz probe.
func StartPortForward(namespace, podName string, localPort, remotePort int, ready func() error) (*PortForward, error) {
	pf := &PortForward{namespace: namespace, podName: podName, localPort: localPort, remotePort: remotePort}
	if err := pf.start(); err != nil {
		return nil, err
	}
	if ready != nil {
		deadline := time.Now().Add(60 * time.Second)
		var lastErr error
		for time.Now().Before(deadline) {
			if lastErr = ready(); lastErr == nil {
				return pf, nil
			}
			time.Sleep(2 * time.Second)
		}
		pf.Stop()
		return nil, fmt.Errorf("port-forward target not ready after 60s: %w", lastErr)
	}
	time.Sleep(2 * time.Second)
	return pf, nil
}

func (p *PortForward) start() error {
	cmd := exec.Command("kubectl", "port-forward",
		"-n", p.namespace,
		p.podName,
		fmt.Sprintf("%d:%d", p.localPort, p.remotePort),
	)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start port-forward: %w", err)
	}
	p.cmd = cmd
	return nil
}

// Stop kills the port-forward subprocess if it is running. Kill/Wait errors
// are logged and swallowed: the subprocess is either already dead (ErrClosed)
// or unkillable, and either way teardown must proceed.
func (p *PortForward) Stop() {
	if p.cmd != nil && p.cmd.Process != nil {
		if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			log.Printf("kill port-forward (ignored): %v", err)
		}
		if _, err := p.cmd.Process.Wait(); err != nil {
			log.Printf("reap port-forward (ignored): %v", err)
		}
	}
	p.cmd = nil
}
