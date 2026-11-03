/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package reset

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	bkeexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
)

func shellCommandFromArgs(args []string) string {
	if len(args) >= 2 && args[0] == "-c" {
		return args[1]
	}
	if len(args) > 0 {
		return args[len(args)-1]
	}
	return ""
}

func applyCommandOutputPatch(patches *gomonkey.Patches, handler func(shellCmd string) (string, error)) {
	patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *bkeexec.CommandExecutor, _ string, args ...string) (string, error) {
			return handler(shellCommandFromArgs(args))
		})
}

func applyCommandCombinedOutputPatch(patches *gomonkey.Patches, handler func(shellCmd string) (string, error)) {
	patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
		func(_ *bkeexec.CommandExecutor, _ string, args ...string) (string, error) {
			return handler(shellCommandFromArgs(args))
		})
}

func TestCleanIpLinkCaliInterface(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandCombinedOutputPatch(patches, func(shellCmd string) (string, error) {
		switch {
		case strings.Contains(shellCmd, "ip link") && strings.Contains(shellCmd, "awk"):
			return "cali1234567890@eth0\n", nil
		case strings.Contains(shellCmd, "ip link set"):
			return "", nil
		case strings.Contains(shellCmd, "ip link del"):
			return "failed", errors.New("delete failed")
		default:
			return "", nil
		}
	})
	cleanIpLink()
}

func TestCleanIpRouteDeleteFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandCombinedOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, "ip route show") {
			return "192.168.1.0/24 via 10.0.0.1 dev boc0\n", nil
		}
		return "failed", errors.New("delete failed")
	})
	cleanIpRoute()
}

func TestCleanIpNeighborDeleteFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandCombinedOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, "ip neigh show") {
			return "192.168.1.1 lladdr aa:bb:cc:dd:ee:ff PERMANENT boc0\n", nil
		}
		return "failed", errors.New("delete failed")
	})
	cleanIpNeighbor()
}

func TestCleanKubeletBinUnitNotLoaded(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandCombinedOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, StopKubeletBin) {
			return "Unit kubelet.service not loaded", errors.New("not loaded")
		}
		return "", nil
	})
	cleanKubeletBin()
}

func TestCleanIptablesRulesSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(exec.LookPath, func(string) (string, error) {
		return "/usr/sbin/iptables", nil
	})
	patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
		func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
			return "", nil
		})
	cleanIptablesRules()
}

func TestDeleteInterfaceIfExistsSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
		func(_ *bkeexec.CommandExecutor, _ string, args ...string) (string, error) {
			shellCmd := ""
			if len(args) >= 2 {
				shellCmd = args[1]
			}
			if strings.Contains(shellCmd, "ip link show") {
				return "state UP", nil
			}
			return "", nil
		})
	patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
			return "", nil
		})
	deleteInterfaceIfExists("docker0")
}

func TestCleanDockerK8sContainersWithPods(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	applyCommandOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, dockerListContainers) {
			return "container-a container-b", nil
		}
		return "", nil
	})
	assert.True(t, cleanDockerK8sContainers())
}

func TestRemoveDockerContainerForceRemoveFallback(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	attempt := 0
	applyCommandOutputPatch(patches, func(string) (string, error) {
		attempt++
		return "failed", errors.New("remove failed")
	})
	killCalled := false
	patches.ApplyFunc(killContainerPid, func(string) { killCalled = true })

	removeDockerContainer("pod-1")
	assert.True(t, killCalled)
	assert.GreaterOrEqual(t, attempt, 3)
}

func TestRemoveContainerdContainerForceRemoveFallback(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	attempt := 0
	applyCommandOutputPatch(patches, func(string) (string, error) {
		attempt++
		return "failed", errors.New("remove failed")
	})
	killCalled := false
	patches.ApplyFunc(killCrictlPodPid, func(string) { killCalled = true })

	removeContainerdContainer("pod-1")
	assert.True(t, killCalled)
	assert.GreaterOrEqual(t, attempt, 3)
}

func TestCleanCrictlContainersWithPods(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, crictlListContainers) {
			return "pod-1 pod-2", nil
		}
		return "failed", errors.New("remove failed")
	})
	cleanCrictlContainers()
}

func TestCleanNerdctlContainersWithPods(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandOutputPatch(patches, func(shellCmd string) (string, error) {
		if strings.Contains(shellCmd, nerdctlListContainers) {
			return "ctr-1", nil
		}
		return "failed", errors.New("remove failed")
	})
	cleanNerdctlContainers()
}

func TestUnmountKubeletDirectoryUnmountError(t *testing.T) {
	kubeletDir := "/var/lib/kubelet"
	mountsContent := kubeletDir + "/pods/test1 " + kubeletDir + "/pods/test1 ext4 rw 0 0\n"

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte(mountsContent), nil
	})
	patches.ApplyFunc(syscall.Unmount, func(string, int) error {
		return errors.New("device busy")
	})

	err := unmountKubeletDirectory(kubeletDir)
	assert.NoError(t, err)
}

func TestRemoveDirNotDirectory(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "bke-reset-file-*")
	require.NoError(t, err)
	require.NoError(t, file.Close())

	err = removeDir(file.Name())
	assert.Error(t, err)
}

func TestRemoveFileGivenDirectory(t *testing.T) {
	dir := t.TempDir()
	err := removeFile(dir)
	assert.Error(t, err)
}

func TestRemoveLocalKubernetesDocker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
	fallbackCalled := false
	patches.ApplyFunc(removeContainerWithProcessFallback, func(string, func(string) bool) {
		fallbackCalled = true
	})

	RemoveLocalKubernetes()
	assert.True(t, fallbackCalled)
}

func TestRemoveLocalKubernetesNoRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
	fallbackCalled := false
	patches.ApplyFunc(removeContainerWithProcessFallback, func(string, func(string) bool) {
		fallbackCalled = true
	})

	RemoveLocalKubernetes()
	assert.False(t, fallbackCalled)
}

func TestCleanNeedDeleteFileRemovePathError(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "missing-dir")

	originalNeedDeleteFile := needDeleteFile
	needDeleteFile = []string{target}
	t.Cleanup(func() { needDeleteFile = originalNeedDeleteFile })

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(removePath, func(string) error {
		return errors.New("remove failed")
	})

	cleanNeedDeleteFile()
}

func TestRepoRemoveCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	applyCommandCombinedOutputPatch(patches, func(string) (string, error) {
		return "failed", errors.New("yum failed")
	})
	err := repoRemove("docker-ce")
	assert.Error(t, err)
}
