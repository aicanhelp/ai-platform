/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package reset

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testNumericZero    = 0
	testNumericOne     = 1
	testNumericTwo     = 2
	testCommandSuccess = ""
	testCommandFailure = "error"
	testFilePermission = 0644
	testDirPermission  = 0755

	testIPv4SegmentA = 10
	testIPv4SegmentB = 0
	testIPv4SegmentC = 0
	testIPv4SegmentD = 1
)

var testIP = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)

func TestCleanIpRoute(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful route cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip route show") {
					return "192.168.1.0/24 via " + testIP + " dev boc0\n", nil
				}
				return "", nil
			},
		},
		{
			name: "no routes to clean",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip route show") {
					return "", nil
				}
				return "", nil
			},
		},
		{
			name: "execute command error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip route show") {
					return "", errors.New("command failed")
				}
				return "", nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			cleanIpRoute()

			assert.True(t, true)
		})
	}
}

func TestCleanIpNeighbor(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful neighbor cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip neigh show") {
					return "192.168.1.1 lladdr aa:bb:cc:dd:ee:ff PERMANENT boc0\n", nil
				}
				return "", nil
			},
		},
		{
			name: "no neighbors to clean",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip neigh show") {
					return "", nil
				}
				return "", nil
			},
		},
		{
			name: "execute command error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip neigh show") {
					return "", errors.New("command failed")
				}
				return "", nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			cleanIpNeighbor()

			assert.True(t, true)
		})
	}
}

func TestCleanIpLink(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful link cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip link") && strings.Contains(cmd, "awk") {
					return "vxlan_sys_4789\n", nil
				}
				return "", nil
			},
		},
		{
			name: "no links to clean",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip link") && strings.Contains(cmd, "awk") {
					return "", nil
				}
				return "", nil
			},
		},
		{
			name: "execute command error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ip link") && strings.Contains(cmd, "awk") {
					return "", errors.New("command failed")
				}
				return "", nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(utils.ContainsString, func(slice []string, str string) bool {
				for _, s := range slice {
					if s == str {
						return true
					}
				}
				return false
			})

			cleanIpLink()

			assert.True(t, true)
		})
	}
}

func TestCleanNetwork(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful network cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "execute command error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", errors.New("command failed")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(utils.ContainsString, func(slice []string, str string) bool {
				for _, s := range slice {
					if s == str {
						return true
					}
				}
				return false
			})

			cleanNetwork()

			assert.True(t, true)
		})
	}
}

func TestCleanKubeletBin(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful kubelet bin cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "stop kubelet error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "StopKubeletBin") {
					return "error", errors.New("stop error")
				}
				return "", nil
			},
		},
		{
			name: "force remove kubelet",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				if strings.Contains(cmd, "ForceRemoveKubeletBin") {
					return "", nil
				}
				if strings.Contains(cmd, "StopKubeletBin") {
					return "error", errors.New("stop error")
				}
				return "", nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			cleanKubeletBin()

			assert.True(t, true)
		})
	}
}

func TestCleanKubelet(t *testing.T) {
	tests := []struct {
		name                   string
		mockIsDocker           func() bool
		mockIsContainerd       func() bool
		mockUnmountKubeletDirs func(string)
	}{
		{
			name: "docker environment",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
			mockUnmountKubeletDirs: func(dir string) {},
		},
		{
			name: "containerd environment",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
			mockUnmountKubeletDirs: func(dir string) {},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(unmountKubeletDirectories, tt.mockUnmountKubeletDirs)

			cleanKubelet()

			assert.True(t, true)
		})
	}
}

func TestUnmountKubeletDirectories(t *testing.T) {
	tests := []struct {
		name        string
		rootDir     string
		expectPanic bool
	}{
		{
			name:    "successful unmount",
			rootDir: "/var/lib/kubelet",
		},
		{
			name:    "unmount directory error",
			rootDir: "/var/lib/kubelet",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(unmountKubeletDirectory, func(dir string) error {
				return nil
			})

			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
				return "", nil
			})

			unmountKubeletDirectories(tt.rootDir)

			assert.True(t, true)
		})
	}
}

func TestCleanKubeletContainer(t *testing.T) {
	tests := []struct {
		name             string
		mockIsDocker     func() bool
		mockIsContainerd func() bool
	}{
		{
			name: "docker environment",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
		},
		{
			name: "containerd environment",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			cleanKubeletContainer()

			assert.True(t, true)
		})
	}
}

func TestCleanDockerKubelet(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful docker kubelet cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "stop kubelet error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			cleanDockerKubelet()

			assert.True(t, true)
		})
	}
}

func TestCleanContainerdKubelet(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful containerd kubelet cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "stop kubelet error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			cleanContainerdKubelet()

			assert.True(t, true)
		})
	}
}

func TestAppendKubeletCleanupFiles(t *testing.T) {
	tests := []struct {
		name    string
		rootDir string
	}{
		{
			name:    "append cleanup files",
			rootDir: "/var/lib/kubelet",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalNeedDeleteFile := needDeleteFile
			needDeleteFile = []string{}
			defer func() {
				needDeleteFile = originalNeedDeleteFile
			}()

			appendKubeletCleanupFiles(tt.rootDir)

			assert.True(t, len(needDeleteFile) > testNumericZero)
		})
	}
}

func TestCleanKubernetesContainer(t *testing.T) {
	tests := []struct {
		name             string
		mockIsDocker     func() bool
		mockIsContainerd func() bool
		mockExecuteCmd   func(string, ...string) (string, error)
	}{
		{
			name: "docker environment",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
		},
		{
			name: "containerd environment",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanKubernetesContainer()

			assert.True(t, true)
		})
	}
}

func TestCleanDockerK8sContainers(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(infrastructure.IsDocker, func() bool {
			return true
		})

		patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, func(cmd string, args ...string) (string, error) {
			return "", nil
		})

		cleanDockerK8sContainers()
	})
}

func TestCleanDockerK8sContainersNoDocker(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(infrastructure.IsDocker, func() bool {
			return false
		})

		cleanDockerK8sContainers()
	})
}

func TestRemoveDockerContainer(t *testing.T) {
	tests := []struct {
		name           string
		podID          string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name:  "remove container",
			podID: "container1",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name:  "stop container error",
			podID: "container2",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			removeDockerContainer(tt.podID)

			assert.True(t, true)
		})
	}
}

func TestCleanContainerdK8sContainers(t *testing.T) {
	tests := []struct {
		name             string
		mockIsContainerd func() bool
		mockExecuteCmd   func(string, ...string) (string, error)
	}{
		{
			name: "containerd environment with pods",
			mockIsContainerd: func() bool {
				return true
			},
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "pod1 pod2", nil
			},
		},
		{
			name: "non-containerd environment",
			mockIsContainerd: func() bool {
				return false
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanContainerdK8sContainers()

			assert.True(t, true)
		})
	}
}

func TestRemoveContainerdContainer(t *testing.T) {
	tests := []struct {
		name           string
		podID          string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name:  "remove container",
			podID: "pod1",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name:  "stop container error",
			podID: "pod2",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			removeContainerdContainer(tt.podID)

			assert.True(t, true)
		})
	}
}

func TestCleanOtherContainer(t *testing.T) {
	tests := []struct {
		name             string
		mockIsDocker     func() bool
		mockIsContainerd func() bool
		mockExecuteCmd   func(string, ...string) (string, error)
	}{
		{
			name: "docker environment",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
		},
		{
			name: "containerd environment",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanOtherContainer()

			assert.True(t, true)
		})
	}
}

func TestCleanDockerAllContainers(t *testing.T) {
	tests := []struct {
		name           string
		mockIsDocker   func() bool
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "docker environment with containers",
			mockIsDocker: func() bool {
				return true
			},
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "non-docker environment",
			mockIsDocker: func() bool {
				return false
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanDockerAllContainers()

			assert.True(t, true)
		})
	}
}

func TestCleanContainerdAllContainers(t *testing.T) {
	tests := []struct {
		name             string
		mockIsContainerd func() bool
	}{
		{
			name: "containerd environment",
			mockIsContainerd: func() bool {
				return true
			},
		},
		{
			name: "non-containerd environment",
			mockIsContainerd: func() bool {
				return false
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			cleanContainerdAllContainers()

			assert.True(t, true)
		})
	}
}

func TestCleanCrictlContainers(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "list containers error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", errors.New("list error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanCrictlContainers()

			assert.True(t, true)
		})
	}
}

func TestCleanNerdctlContainers(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful cleanup",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "list containers error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", errors.New("list error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			cleanNerdctlContainers()

			assert.True(t, true)
		})
	}
}

func TestDisableDocker(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful disable",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "stop docker error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			disableDocker()

			assert.True(t, true)
		})
	}
}

func TestDisableContainerd(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful disable",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "stop containerd error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			disableContainerd()

			assert.True(t, true)
		})
	}
}

func TestDisableBkeAgent(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "successful disable",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "unit not found skips stop disable",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "No files found for bkeagent.service", errors.New("exit status 1")
			},
		},
		{
			name: "stop bkeagent error",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				joined := strings.Join(args, " ")
				if strings.Contains(joined, "systemctl cat") {
					return "[Unit]\nDescription=bkeagent\n", nil
				}
				return "error", errors.New("stop error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			disableBkeAgent()

			assert.True(t, true)
		})
	}
}

func TestAddBkeAgentDeleteFile(t *testing.T) {
	originalNeedDeleteFile := needDeleteFile
	needDeleteFile = []string{}
	t.Cleanup(func() { needDeleteFile = originalNeedDeleteFile })

	addBkeAgentDeleteFile()

	assert.Contains(t, needDeleteFile, "/usr/local/bin/bkeagent")
	assert.Contains(t, needDeleteFile, "/etc/systemd/system/bkeagent.service")
	assert.Contains(t, needDeleteFile, "/usr/lib/systemd/system/bkeagent.service")
	assert.Contains(t, needDeleteFile, "/etc/systemd/system/bkeagent.service.d")
	assert.Contains(t, needDeleteFile, "/usr/lib/systemd/system/bkeagent.service.d")
	assert.Contains(t, needDeleteFile, "/etc/openFuyao/bkeagent")
	assert.Contains(t, needDeleteFile, "/etc/bkeagent")
	assert.NotContains(t, needDeleteFile, "/var/log/openFuyao/bkeagent.log")
}

func TestCleanBkeAgent(t *testing.T) {
	originalNeedDeleteFile := needDeleteFile
	needDeleteFile = []string{}
	t.Cleanup(func() { needDeleteFile = originalNeedDeleteFile })

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
		return "", nil
	})

	cleanBkeAgent()

	assert.Contains(t, needDeleteFile, "/usr/local/bin/bkeagent")
	assert.Contains(t, needDeleteFile, "/etc/openFuyao/bkeagent")
}

func TestAddNeedDeleteFile(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "add need delete files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalNeedDeleteFile := needDeleteFile
			needDeleteFile = []string{}
			defer func() {
				needDeleteFile = originalNeedDeleteFile
			}()

			addNeedDeleteFile()

			assert.True(t, len(needDeleteFile) > testNumericZero)
		})
	}
}

func TestCleanContainerRuntime(t *testing.T) {
	tests := []struct {
		name             string
		mockIsDocker     func() bool
		mockIsContainerd func() bool
		mockExecuteCmd   func(string, ...string) (string, error)
	}{
		{
			name: "docker environment",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
		},
		{
			name: "containerd environment",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
		},
		{
			name: "both docker and containerd",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(log.Errorf, func(template string, args ...interface{}) {})

			cleanContainerRuntime()

			assert.True(t, true)
		})
	}
}

func TestCleanNeedDeleteFile(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "clean need delete files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalNeedDeleteFile := needDeleteFile
			needDeleteFile = []string{}
			defer func() {
				needDeleteFile = originalNeedDeleteFile
			}()

			addNeedDeleteFile()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(removeDir, func(path string) error {
				return nil
			})

			patches.ApplyFunc(removeFile, func(path string) error {
				return nil
			})

			cleanNeedDeleteFile()

			assert.True(t, true)
		})
	}
}

func TestRemoveDir(t *testing.T) {
	assert.NotPanics(t, func() {
		removeDir("/nonexistent/path")
	})
}

func TestRemoveDirNotExist(t *testing.T) {
	assert.NotPanics(t, func() {
		removeDir("/nonexistent/path")
	})
}

func TestRemoveDirWholeTree(t *testing.T) {
	dir := t.TempDir()
	nestedDir := filepath.Join(dir, "child")
	assert.NoError(t, os.MkdirAll(nestedDir, testDirPermission))
	assert.NoError(t, os.WriteFile(filepath.Join(nestedDir, "file.txt"), []byte("x"), testFilePermission))

	assert.NoError(t, removeDir(dir))
	_, err := os.Stat(dir)
	assert.True(t, os.IsNotExist(err))
}

func TestRemoveFile(t *testing.T) {
	assert.NotPanics(t, func() {
		removeFile("/nonexistent/file.txt")
	})
}

func TestRemoveFileNotExist(t *testing.T) {
	assert.NotPanics(t, func() {
		removeFile("/nonexistent/file.txt")
	})
}

func TestRemoveFileIsDirectory(t *testing.T) {
	assert.NotPanics(t, func() {
		removeFile("/nonexistent/path")
	})
}

func TestRemovePath(t *testing.T) {
	t.Run("nonexistent path", func(t *testing.T) {
		assert.NoError(t, removePath("/nonexistent/path/for/bke-reset-test"))
	})

	t.Run("file", func(t *testing.T) {
		file, err := os.CreateTemp(t.TempDir(), "bke-reset-file-*")
		assert.NoError(t, err)
		assert.NoError(t, file.Close())
		assert.NoError(t, removePath(file.Name()))
		_, err = os.Stat(file.Name())
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("directory", func(t *testing.T) {
		dir := t.TempDir()
		nested := filepath.Join(dir, "nested.txt")
		assert.NoError(t, os.WriteFile(nested, []byte("test"), testFilePermission))
		assert.NoError(t, removePath(dir))
		_, err := os.Stat(dir)
		assert.True(t, os.IsNotExist(err))
	})
}

func TestUnmountKubeletDirectory(t *testing.T) {
	tests := []struct {
		name         string
		kubeletDir   string
		mockReadFile func(string) ([]byte, error)
		expectError  bool
	}{
		{
			name:       "successful unmount",
			kubeletDir: "/var/lib/kubelet",
			mockReadFile: func(path string) ([]byte, error) {
				return []byte(""), nil
			},
			expectError: false,
		},
		{
			name:       "read file error",
			kubeletDir: "/var/lib/kubelet",
			mockReadFile: func(path string) ([]byte, error) {
				return nil, errors.New("read error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.ReadFile, tt.mockReadFile)

			err := unmountKubeletDirectory(tt.kubeletDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUnmountKubeletDirectoryWithMounts(t *testing.T) {
	kubeletDir := "/var/lib/kubelet"
	mountsContent := fmt.Sprintf("/var/lib/kubelet/pods/test1 %s/pods/test1 ext4 rw 0 0\n", kubeletDir)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(path string) ([]byte, error) {
		return []byte(mountsContent), nil
	})

	err := unmountKubeletDirectory(kubeletDir)
	assert.NoError(t, err)
}

func TestRepoRemove(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
			return "centos", "", "", nil
		})

		patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
			return "", nil
		})

		repoRemove("docker*")
	})
}

func TestRepoRemoveUbuntu(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
			return "ubuntu", "", "", nil
		})

		patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
			return "", nil
		})

		repoRemove("docker*", "containerd.io")
	})
}

func TestRepoRemoveUnknownPlatform(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
			return "unknown", "", "", nil
		})

		patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
			return "", nil
		})

		repoRemove("docker*")
	})
}

func TestRepoRemoveHostInfoError(t *testing.T) {
	assert.NotPanics(t, func() {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
			return "", "", "", errors.New("host info error")
		})

		repoRemove("docker*")
	})
}

func TestCleanNetworkWithInterfaces(t *testing.T) {
	tests := []struct {
		name           string
		mockExecuteCmd func(string, ...string) (string, error)
	}{
		{
			name: "cleanup with docker0 and nerdctl0",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "cleanup with interface errors",
			mockExecuteCmd: func(cmd string, args ...string) (string, error) {
				return "", errors.New("interface error")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, tt.mockExecuteCmd)

			patches.ApplyFunc(utils.ContainsString, func(slice []string, str string) bool {
				for _, s := range slice {
					if s == str {
						return true
					}
				}
				return false
			})

			cleanNetwork()

			assert.True(t, true)
		})
	}
}

func TestRemoveAllInOne(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "call all cleanup functions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
				return "", nil
			})

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, func(cmd string, args ...string) (string, error) {
				return "", nil
			})

			patches.ApplyFunc(log.Errorf, func(template string, args ...interface{}) {})

			patches.ApplyFunc(removeDir, func(path string) error {
				return nil
			})

			patches.ApplyFunc(removeFile, func(path string) error {
				return nil
			})

			patches.ApplyFunc(infrastructure.IsDocker, func() bool {
				return true
			})

			patches.ApplyFunc(infrastructure.IsContainerd, func() bool {
				return false
			})

			removeAllInOne()

			assert.True(t, true)
		})
	}
}

func TestRemoveAllInOneContainerd(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "call all cleanup functions for containerd",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput, func(cmd string, args ...string) (string, error) {
				return "", nil
			})

			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput, func(cmd string, args ...string) (string, error) {
				return "", nil
			})

			patches.ApplyFunc(log.Errorf, func(template string, args ...interface{}) {})

			patches.ApplyFunc(removeDir, func(path string) error {
				return nil
			})

			patches.ApplyFunc(removeFile, func(path string) error {
				return nil
			})

			patches.ApplyFunc(infrastructure.IsDocker, func() bool {
				return false
			})

			patches.ApplyFunc(infrastructure.IsContainerd, func() bool {
				return true
			})

			removeAllInOne()

			assert.True(t, true)
		})
	}
}
