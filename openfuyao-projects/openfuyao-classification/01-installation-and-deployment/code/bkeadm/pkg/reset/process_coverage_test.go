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
	"fmt"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/shirou/gopsutil/v3/process"
	"github.com/stretchr/testify/assert"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	bkeexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
)

func TestRemoveContainerWithProcessFallbackContainerNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	t.Cleanup(func() { global.Docker = originalDocker })

	patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
		return types.ContainerJSON{}, false
	})

	removeContainerWithProcessFallback("missing", isResidualKubeProcess)
}

func TestRemoveContainerWithProcessFallbackDockerRemoved(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	t.Cleanup(func() { global.Docker = originalDocker })

	removed := false
	patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
		if removed {
			return types.ContainerJSON{}, false
		}
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{Running: true, Pid: 1234},
			},
		}, true
	})
	patches.ApplyFunc((*docker.Client).ContainerRemove, func(*docker.Client, string) error {
		removed = true
		return nil
	})

	removeContainerWithProcessFallback("local-k8s", isResidualK3sProcess)
}

func TestRemoveContainerWithProcessFallbackDockerRetryCleanup(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	t.Cleanup(func() { global.Docker = originalDocker })

	stillExists := true
	patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
		if !stillExists {
			return types.ContainerJSON{}, false
		}
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{Running: true, Pid: 4321},
			},
		}, true
	})
	patches.ApplyFunc((*docker.Client).ContainerRemove, func(*docker.Client, string) error {
		return fmt.Errorf("remove failed")
	})
	killPidCalled := false
	patches.ApplyFunc(killContainerPid, func(string) { killPidCalled = true })
	killResidualCalled := false
	patches.ApplyFunc(killResidualProcesses, func(func(string) bool) { killResidualCalled = true })

	removeContainerWithProcessFallback("stuck-container", isResidualKubeProcess)
	assert.True(t, killPidCalled)
	assert.True(t, killResidualCalled)
}

func TestRemoveContainerWithProcessFallbackContainerdRemoved(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return true })

	removed := false
	patches.ApplyFunc(econd.ContainerExists, func(string) (econd.NerdContainerInfo, bool) {
		if removed {
			return econd.NerdContainerInfo{}, false
		}
		info := econd.NerdContainerInfo{}
		info.State.Pid = 5678
		return info, true
	})
	patches.ApplyFunc(econd.ContainerRemove, func(string) error {
		removed = true
		return nil
	})

	removeContainerWithProcessFallback("local-k8s", isResidualK3sProcess)
}

func TestRemoveContainerWithProcessFallbackContainerdRetry(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return true })

	patches.ApplyFunc(econd.ContainerExists, func(string) (econd.NerdContainerInfo, bool) {
		info := econd.NerdContainerInfo{}
		info.State.Pid = 9012
		return info, true
	})
	patches.ApplyFunc(econd.ContainerRemove, func(string) error {
		return errors.New("remove failed")
	})
	patches.ApplyFunc(killContainerPid, func(string) {})
	patches.ApplyFunc(killResidualProcesses, func(func(string) bool) {})

	removeContainerWithProcessFallback("stuck-container", isResidualKubeProcess)
}

func TestKillContainerPidDocker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	t.Cleanup(func() { global.Docker = originalDocker })

	patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{Pid: 2468},
			},
		}, true
	})
	treeCalled := false
	patches.ApplyFunc(killProcessTree, func(int32) { treeCalled = true })

	killContainerPid("kube-container")
	assert.True(t, treeCalled)
}

func TestKillContainerPidDockerNoPid(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	t.Cleanup(func() { global.Docker = originalDocker })

	patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{Pid: 0},
			},
		}, true
	})

	killContainerPid("kube-container")
}

func TestKillContainerPidContainerd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return true })

	patches.ApplyFunc(econd.ContainerExists, func(string) (econd.NerdContainerInfo, bool) {
		info := econd.NerdContainerInfo{}
		info.State.Pid = 1357
		return info, true
	})
	treeCalled := false
	patches.ApplyFunc(killProcessTree, func(int32) { treeCalled = true })

	killContainerPid("kube-container")
	assert.True(t, treeCalled)
}

func TestKillContainerPidNoRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	killContainerPid("kube-container")
}

func TestKillCrictlPodPid(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
			func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
				return "", nil
			})
		killCrictlPodPid("pod-abc")
	})

	t.Run("command fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
			func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
				return "permission denied", errors.New("command failed")
			})
		killCrictlPodPid("pod-abc")
	})
}

func TestKillResidualProcesses(t *testing.T) {
	t.Run("list pids fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(process.Pids, func() ([]int32, error) {
			return nil, errors.New("list failed")
		})
		killResidualProcesses(isResidualKubeProcess)
	})

	t.Run("kills matching process", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(process.Pids, func() ([]int32, error) {
			return []int32{1001, 1002}, nil
		})
		patches.ApplyFunc(process.NewProcess, func(pid int32) (*process.Process, error) {
			return &process.Process{Pid: pid}, nil
		})
		patches.ApplyFunc((*process.Process).Cmdline, func(p *process.Process) (string, error) {
			if p.Pid == 1001 {
				return "/usr/bin/kubelet --config=/var/lib/kubelet/config.yaml", nil
			}
			return "", nil
		})
		patches.ApplyFunc(syscall.Kill, func(int, syscall.Signal) error {
			return nil
		})
		killResidualProcesses(isResidualKubeProcess)
	})

	t.Run("skip new process error and empty cmdline", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(process.Pids, func() ([]int32, error) {
			return []int32{2001, 2002}, nil
		})
		patches.ApplyFunc(process.NewProcess, func(pid int32) (*process.Process, error) {
			if pid == 2001 {
				return nil, errors.New("process gone")
			}
			return &process.Process{Pid: pid}, nil
		})
		patches.ApplyFunc((*process.Process).Cmdline, func(*process.Process) (string, error) {
			return "", errors.New("access denied")
		})
		killResidualProcesses(isResidualKubeProcess)
	})

	t.Run("kill syscall fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(process.Pids, func() ([]int32, error) {
			return []int32{3001}, nil
		})
		patches.ApplyFunc(process.NewProcess, func(pid int32) (*process.Process, error) {
			return &process.Process{Pid: pid}, nil
		})
		patches.ApplyFunc((*process.Process).Cmdline, func(*process.Process) (string, error) {
			return "/usr/local/bin/etcd --data-dir=/var/lib/etcd", nil
		})
		patches.ApplyFunc(syscall.Kill, func(int, syscall.Signal) error {
			return errors.New("operation not permitted")
		})
		killResidualProcesses(isResidualKubeProcess)
	})
}

func TestKillProcessTreeSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(process.NewProcess, func(pid int32) (*process.Process, error) {
		return &process.Process{Pid: pid}, nil
	})
	patches.ApplyFunc((*process.Process).Children, func(p *process.Process) ([]*process.Process, error) {
		if p.Pid == 100 {
			return []*process.Process{{Pid: 42}}, nil
		}
		return nil, nil
	})
	childKilled := false
	patches.ApplyFunc((*process.Process).Kill, func(p *process.Process) error {
		if p.Pid == 42 {
			childKilled = true
		}
		return nil
	})

	killProcessTree(100)
	assert.True(t, childKilled)
}

func TestKillProcessTreeKillFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(process.NewProcess, func(pid int32) (*process.Process, error) {
		return &process.Process{Pid: pid}, nil
	})
	patches.ApplyFunc((*process.Process).Children, func(*process.Process) ([]*process.Process, error) {
		return nil, errors.New("no children")
	})
	patches.ApplyFunc((*process.Process).Kill, func(*process.Process) error {
		return errors.New("kill failed")
	})

	killProcessTree(100)
}

func TestKillResidualKubeProcesses(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(process.Pids, func() ([]int32, error) {
		return []int32{}, nil
	})
	killResidualKubeProcesses()
}
