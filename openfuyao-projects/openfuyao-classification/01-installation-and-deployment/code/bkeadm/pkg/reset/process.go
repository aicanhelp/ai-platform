/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package reset

import (
	"fmt"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/process"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const removeContainerMaxRetry = 2

// killResidualKubeProcesses is a best-effort fallback to avoid leaving kube-related
// host processes around after reset (e.g. when container removal partially fails).
func killResidualKubeProcesses() {
	killResidualProcesses(isResidualKubeProcess)
}

// removeContainerWithProcessFallback removes a container; on failure, kills residual
// processes (by container PID and optional cmdline matcher) then retries.
func removeContainerWithProcessFallback(name string, isResidualProcess func(cmdline string) bool) {
	removeFn := func() error {
		if infrastructure.IsDocker() {
			return global.Docker.ContainerRemove(name)
		}
		if infrastructure.IsContainerd() {
			return econd.ContainerRemove(name)
		}
		return nil
	}
	existsFn := func() bool {
		if infrastructure.IsDocker() {
			_, exist := global.Docker.ContainerExists(name)
			return exist
		}
		if infrastructure.IsContainerd() {
			_, exist := econd.ContainerExists(name)
			return exist
		}
		return false
	}

	if !existsFn() {
		return
	}

	for i := 0; i < removeContainerMaxRetry; i++ {
		err := removeFn()
		if err != nil {
			log.Warnf("remove container %s failed: %v", name, err)
		}
		if !existsFn() {
			return
		}
		log.Warnf("container %s still exists after remove, cleaning residual processes as fallback", name)
		killContainerPid(name)
		if isResidualProcess != nil {
			killResidualProcesses(isResidualProcess)
		}
		time.Sleep(utils.ContainerRemoveWaitSeconds * time.Second)
	}

	if err := removeFn(); err != nil {
		log.Warnf("remove container %s still failed after residual process cleanup: %v", name, err)
	}
	if existsFn() {
		log.Warnf("container %s still exists after residual process cleanup", name)
	}
}

// killContainerPid kills the main process (and its children) of a container via inspect.
func killContainerPid(name string) {
	var pid int32
	if infrastructure.IsDocker() {
		info, exist := global.Docker.ContainerExists(name)
		if !exist || info.State == nil || info.State.Pid == 0 {
			return
		}
		pid = int32(info.State.Pid)
	} else if infrastructure.IsContainerd() {
		info, exist := econd.ContainerExists(name)
		if !exist || info.State.Pid == 0 {
			return
		}
		pid = int32(info.State.Pid)
	} else {
		return
	}
	log.Warnf("killing container %s process tree pid=%d", name, pid)
	killProcessTree(pid)
}

// killCrictlPodPid kills residual processes of a crictl pod by inspecting its pid.
func killCrictlPodPid(podID string) {
	cmd := fmt.Sprintf(
		`pid=$(crictl inspectp %s -o go-template='{{.info.pid}}' 2>/dev/null); `+
			`[ -n "$pid" ] && [ "$pid" != "0" ] && kill -9 "$pid"`, podID)
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("kill crictl pod %s pid failed: %v, output: %s", podID, err, out)
	}
}

// killResidualProcesses kills host processes whose cmdline matches the given predicate.
func killResidualProcesses(match func(cmdline string) bool) {
	pids, err := process.Pids()
	if err != nil {
		log.Warnf("list processes failed: %v", err)
		return
	}
	for _, pid := range pids {
		pn, err := process.NewProcess(pid)
		if err != nil {
			continue
		}
		cmd, err := pn.Cmdline()
		if err != nil || cmd == "" {
			continue
		}
		if !match(cmd) {
			continue
		}
		log.Warnf("killing residual process pid=%d cmd=%s", pid, cmd)
		if err := syscall.Kill(int(pid), syscall.SIGKILL); err != nil {
			log.Warnf("kill process %d failed: %v", pid, err)
		}
	}
}

func killProcessTree(pid int32) {
	p, err := process.NewProcess(pid)
	if err != nil {
		return
	}
	children, err := p.Children()
	if err == nil {
		for _, child := range children {
			killProcessTree(child.Pid)
		}
	}
	if err := p.Kill(); err != nil {
		log.Warnf("kill process %d failed: %v", pid, err)
	}
}

// isResidualK3sProcess returns true for leftover k3s processes from the local kubernetes container.
func isResidualK3sProcess(cmdline string) bool {
	if strings.Contains(cmdline, "bke ") {
		return false
	}
	return strings.Contains(cmdline, "k3s server") ||
		strings.Contains(cmdline, "k3s-server") ||
		strings.Contains(cmdline, "/bin/k3s") ||
		strings.Contains(cmdline, "/usr/local/bin/k3s")
}

// isResidualKubeProcess matches kube-related processes that commonly linger after
// failed/partial teardown. It reuses isResidualK3sProcess for local kubernetes leftovers
// and additionally matches core kubernetes components.
func isResidualKubeProcess(cmdline string) bool {
	if cmdline == "" {
		return false
	}
	if isResidualK3sProcess(cmdline) {
		return true
	}
	c := strings.ToLower(cmdline)
	needles := []string{
		"kubelet",
		"kube-apiserver",
		"kube-controller-manager",
		"kube-scheduler",
		"kube-proxy",
		"etcd",
	}
	for _, n := range needles {
		if strings.Contains(c, n) {
			return true
		}
	}
	return false
}
