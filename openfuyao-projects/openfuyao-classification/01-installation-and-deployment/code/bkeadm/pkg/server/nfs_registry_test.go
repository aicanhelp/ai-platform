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

package server

import (
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/network"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
)

const (
	testNFSName        = "test-nfs"
	testNFSImage       = "itsthenetwork/nfs-ganesha:latest"
	testNFSDataDir     = "/var/nfs"
	testNFSPort        = "2049"
	testContainerCount = 2
)

func TestStartNFSServerWithDocker(t *testing.T) {
	tests := []struct {
		name                        string
		mockIsDocker                func() bool
		mockIsContainerd            func() bool
		mockEnsureImageAndContainer func(name, image, serviceName string) (bool, error)
		mockRunDockerNFSServer      func(string, string, string) error
		expectError                 bool
	}{
		{
			name: "successful start with docker",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
			mockEnsureImageAndContainer: func(name, image, serviceName string) (bool, error) {
				return false, nil
			},
			mockRunDockerNFSServer: func(name, image, nfsDataDirectory string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "nfs already running",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
			mockEnsureImageAndContainer: func(name, image, serviceName string) (bool, error) {
				return true, nil
			},
			expectError: false,
		},
		{
			name: "ensure image and container fails",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
			mockEnsureImageAndContainer: func(name, image, serviceName string) (bool, error) {
				return false, assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			if tt.mockEnsureImageAndContainer != nil {
				patches.ApplyFunc(ensureDockerImageAndContainer, tt.mockEnsureImageAndContainer)
			}

			if tt.mockRunDockerNFSServer != nil {
				patches.ApplyFunc(runDockerNFSServer, tt.mockRunDockerNFSServer)
			}

			patches.ApplyFunc(waitForDockerContainerRunning, func(name, serviceType string) {})

			err := StartNFSServer(testNFSName, testNFSImage, testNFSDataDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartNFSServerWithContainerd(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	tests := []struct {
		name                             string
		mockIsDocker                     func() bool
		mockIsContainerd                 func() bool
		mockContainerdEnsureImageExists  func(string) error
		mockContainerdEnsureContainerRun func(string) (bool, error)
		mockContainerdRun                func([]string) error
		mockContainerdInspect            func(string) (containerd.NerdContainerInfo, error)
		expectError                      bool
	}{
		{
			name: "successful start with containerd",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
			mockContainerdEnsureImageExists: func(image string) error {
				return nil
			},
			mockContainerdEnsureContainerRun: func(name string) (bool, error) {
				return false, nil
			},
			mockContainerdRun: func(script []string) error {
				return nil
			},
			mockContainerdInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{
					State: struct {
						Status     string `json:"Status"`
						Running    bool   `json:"Running"`
						Paused     bool   `json:"Paused"`
						Restarting bool   `json:"Restarting"`
						Pid        uint   `json:"Pid"`
						ExitCode   uint   `json:"ExitCode"`
						FinishedAt string `json:"FinishedAt"`
					}{
						Running: true,
					},
				}, nil
			},
			expectError: false,
		},
		{
			name: "containerd container already running",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
			mockContainerdEnsureImageExists: func(image string) error {
				return nil
			},
			mockContainerdEnsureContainerRun: func(name string) (bool, error) {
				return true, nil
			},
			expectError: false,
		},
		{
			name: "containerd ensure image exists fails",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
			mockContainerdEnsureImageExists: func(image string) error {
				return assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			if tt.mockContainerdEnsureImageExists != nil {
				patches.ApplyFunc(containerd.EnsureImageExists, tt.mockContainerdEnsureImageExists)
			}

			if tt.mockContainerdEnsureContainerRun != nil {
				patches.ApplyFunc(containerd.EnsureContainerRun, tt.mockContainerdEnsureContainerRun)
			}

			if tt.mockContainerdRun != nil {
				patches.ApplyFunc(containerd.Run, tt.mockContainerdRun)
			}

			if tt.mockContainerdInspect != nil {
				patches.ApplyFunc(containerd.ContainerInspect, tt.mockContainerdInspect)
			}

			err := StartNFSServer(testNFSName, testNFSImage, testNFSDataDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartNFSServerNoRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool {
		return false
	})

	patches.ApplyFunc(infrastructure.IsContainerd, func() bool {
		return false
	})

	err := StartNFSServer(testNFSName, testNFSImage, testNFSDataDir)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "docker or containerd runtime not found")
}

func TestRemoveNFSServer(t *testing.T) {
	tests := []struct {
		name                string
		mockIsDocker        func() bool
		mockIsContainerd    func() bool
		mockRemoveContainer func(*docker.Client, string) error
		mockContainerExists func(*docker.Client, string) (types.ContainerJSON, bool)
		mockExecuteCommand  func(string, ...string) error
		expectError         bool
	}{
		{
			name: "successful removal with docker",
			mockIsDocker: func() bool {
				return true
			},
			mockIsContainerd: func() bool {
				return false
			},
			mockRemoveContainer: func(c *docker.Client, name string) error {
				return nil
			},
			mockContainerExists: func(c *docker.Client, name string) (types.ContainerJSON, bool) {
				return types.ContainerJSON{}, false
			},
			mockExecuteCommand: func(command string, args ...string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "successful removal with containerd",
			mockIsDocker: func() bool {
				return false
			},
			mockIsContainerd: func() bool {
				return true
			},
			mockRemoveContainer: func(c *docker.Client, name string) error {
				return nil
			},
			mockContainerExists: func(c *docker.Client, name string) (types.ContainerJSON, bool) {
				return types.ContainerJSON{}, false
			},
			mockExecuteCommand: func(command string, args ...string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalDocker := global.Docker
			global.Docker = &docker.Client{}
			defer func() {
				global.Docker = originalDocker
			}()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			if tt.mockRemoveContainer != nil {
				patches.ApplyFunc((*docker.Client).ContainerRemove, tt.mockRemoveContainer)
			}

			if tt.mockContainerExists != nil {
				patches.ApplyFunc((*docker.Client).ContainerExists, tt.mockContainerExists)
			}

			if tt.mockExecuteCommand != nil {
				patches.ApplyFunc(global.Command.ExecuteCommand, tt.mockExecuteCommand)
			}

			err := RemoveNFSServer(testNFSName)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRunDockerNFSServer(t *testing.T) {
	tests := []struct {
		name        string
		mockRun     func(*docker.Client, *container.Config, *container.HostConfig, *network.NetworkingConfig, *specs.Platform, string) error
		expectError bool
	}{
		{
			name: "successful docker run",
			mockRun: func(c *docker.Client, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, name string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "docker run fails",
			mockRun: func(c *docker.Client, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, name string) error {
				return assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalDocker := global.Docker
			global.Docker = &docker.Client{}
			defer func() {
				global.Docker = originalDocker
			}()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*docker.Client).Run, tt.mockRun)

			err := runDockerNFSServer(testNFSName, testNFSImage, testNFSDataDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartNFSServerWithContainerdNotRunning(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	inspectCallCount := 0

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(infrastructure.IsDocker, func() bool {
		return false
	})

	patches.ApplyFunc(infrastructure.IsContainerd, func() bool {
		return true
	})

	patches.ApplyFunc(containerd.EnsureImageExists, func(image string) error {
		return nil
	})

	patches.ApplyFunc(containerd.EnsureContainerRun, func(name string) (bool, error) {
		return false, nil
	})

	patches.ApplyFunc(containerd.Run, func(script []string) error {
		return nil
	})

	patches.ApplyFunc(containerd.ContainerInspect, func(name string) (containerd.NerdContainerInfo, error) {
		inspectCallCount++
		if inspectCallCount < testContainerCount {
			return containerd.NerdContainerInfo{
				State: struct {
					Status     string `json:"Status"`
					Running    bool   `json:"Running"`
					Paused     bool   `json:"Paused"`
					Restarting bool   `json:"Restarting"`
					Pid        uint   `json:"Pid"`
					ExitCode   uint   `json:"ExitCode"`
					FinishedAt string `json:"FinishedAt"`
				}{
					Running: false,
				},
			}, nil
		}
		return containerd.NerdContainerInfo{
			State: struct {
				Status     string `json:"Status"`
				Running    bool   `json:"Running"`
				Paused     bool   `json:"Paused"`
				Restarting bool   `json:"Restarting"`
				Pid        uint   `json:"Pid"`
				ExitCode   uint   `json:"ExitCode"`
				FinishedAt string `json:"FinishedAt"`
			}{
				Running: true,
			},
		}, nil
	})

	err := StartNFSServer(testNFSName, testNFSImage, testNFSDataDir)

	assert.NoError(t, err)
}
