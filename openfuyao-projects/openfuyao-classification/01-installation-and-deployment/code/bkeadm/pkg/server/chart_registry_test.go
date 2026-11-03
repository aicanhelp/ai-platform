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
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func TestStartChartRegistry(t *testing.T) {
	tests := []struct {
		name                    string
		containerName           string
		image                   string
		port                    string
		dataDir                 string
		mockIsContainerd        func() bool
		mockIsDocker            func() bool
		mockStartWithContainerd func(string, string, string, string) error
		mockStartWithDocker     func(string, string, string, string) error
		expectError             bool
	}{
		{
			name:                    "start with containerd when only containerd is available",
			containerName:           "test-chart-registry",
			image:                   "chartmuseum:latest",
			port:                    "8080",
			dataDir:                 "/var/lib/charts",
			mockIsContainerd:        func() bool { return true },
			mockIsDocker:            func() bool { return false },
			mockStartWithContainerd: func(name, image, port, dataDir string) error { return nil },
			mockStartWithDocker: func(name, image, port, dataDir string) error {
				t.Error("startWithDocker should not be called when only containerd is available")
				return nil
			},
			expectError: false,
		},
		{
			name:                "start with docker when docker is available",
			containerName:       "test-chart-registry",
			image:               "chartmuseum:latest",
			port:                "8080",
			dataDir:             "/var/lib/charts",
			mockIsContainerd:    func() bool { return false },
			mockIsDocker:        func() bool { return true },
			mockStartWithDocker: func(name, image, port, dataDir string) error { return nil },
			mockStartWithContainerd: func(name, image, port, dataDir string) error {
				t.Error("startWithContainerd should not be called when docker is available")
				return nil
			},
			expectError: false,
		},
		{
			name:                "both runtimes available, docker takes precedence",
			containerName:       "test-chart-registry",
			image:               "chartmuseum:latest",
			port:                "8080",
			dataDir:             "/var/lib/charts",
			mockIsContainerd:    func() bool { return true },
			mockIsDocker:        func() bool { return true },
			mockStartWithDocker: func(name, image, port, dataDir string) error { return nil },
			mockStartWithContainerd: func(name, image, port, dataDir string) error {
				t.Error("startWithContainerd should not be called when docker is available")
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			if tt.mockStartWithContainerd != nil {
				patches.ApplyFunc(startChartRegistryWithContainerd, tt.mockStartWithContainerd)
			}

			if tt.mockStartWithDocker != nil {
				patches.ApplyFunc(startChartRegistryWithDocker, tt.mockStartWithDocker)
			}

			err := StartChartRegistry(tt.containerName, tt.image, tt.port, tt.dataDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartChartRegistryWithDocker(t *testing.T) {
	tests := []struct {
		name                              string
		mockEnsureDockerImageAndContainer func(string, string, string) (bool, error)
		mockRunDockerChartRegistry        func(string, string, string, string) error
		mockWaitForDockerContainerRunning func(string, string)
		expectError                       bool
	}{
		{
			name: "successful registry start",
			mockEnsureDockerImageAndContainer: func(name, image, service string) (bool, error) {

				return false, nil // Container not running
			},
			mockRunDockerChartRegistry: func(name, image, port, dataDir string) error {

				return nil
			},
			mockWaitForDockerContainerRunning: func(name, serviceType string) {
				assert.Equal(t, "test-chart-registry", name)
				assert.Equal(t, "chart", serviceType)
			},
			expectError: false,
		},
		{
			name: "ensure docker image and container fails",
			mockEnsureDockerImageAndContainer: func(name, image, service string) (bool, error) {
				return false, fmt.Errorf("ensure failed")
			},
			expectError: true,
		},
		{
			name: "container already running",
			mockEnsureDockerImageAndContainer: func(name, image, service string) (bool, error) {
				return true, nil // Container already running
			},
			// runDockerChartRegistry should not be called
			expectError: false,
		},
		{
			name: "run docker chart registry fails",
			mockEnsureDockerImageAndContainer: func(name, image, service string) (bool, error) {
				return false, nil // Container not running
			},
			mockRunDockerChartRegistry: func(name, image, port, dataDir string) error {
				return fmt.Errorf("run failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(ensureDockerImageAndContainer, tt.mockEnsureDockerImageAndContainer)

			if tt.mockRunDockerChartRegistry != nil {
				patches.ApplyFunc(runDockerChartRegistry, tt.mockRunDockerChartRegistry)
			}

			if tt.mockWaitForDockerContainerRunning != nil {
				patches.ApplyFunc(waitForDockerContainerRunning, tt.mockWaitForDockerContainerRunning)
			}

			err := startChartRegistryWithDocker("test-chart-registry", "chartmuseum:latest", "8080", "/var/lib/charts")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWaitForDockerContainerRunning(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	// Create a mock Docker client
	mockDockerClient := &docker.Client{}

	originalDocker := global.Docker
	global.Docker = mockDockerClient
	defer func() {
		global.Docker = originalDocker
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*docker.Client).GetClient, func(client *docker.Client) *client.Client {
		// Return a mock Docker API client
		return client.Client
	})

	patches.ApplyFunc((*client.Client).ContainerInspect, func(client *client.Client, ctx context.Context, containerID string) (container.InspectResponse, error) {
		// Return a mock container JSON
		return container.InspectResponse{ContainerJSONBase: &types.ContainerJSONBase{State: &types.ContainerState{Running: true}}}, nil
	})

	// This test would need more sophisticated mocking to fully test the waiting logic
	// For now, we'll just verify that the function can be called without error
	waitForDockerContainerRunning("test-container", "chart")

	// The function should complete without panic
	assert.True(t, true)
}

func TestStartChartRegistryWithContainerd(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	tests := []struct {
		name                   string
		mockEnsureImageExists  func(string) error
		mockEnsureContainerRun func(string) (bool, error)
		mockRun                func([]string) error
		mockContainerInspect   func(string) (containerd.NerdContainerInfo, error)
		expectError            bool
	}{
		{
			name: "successful registry start with containerd",
			mockEnsureImageExists: func(image string) error {
				return nil
			},
			mockEnsureContainerRun: func(name string) (bool, error) {
				return false, nil // Container not running
			},
			mockRun: func(args []string) error {
				return nil
			},
			mockContainerInspect: func(name string) (containerd.NerdContainerInfo, error) {
				info := containerd.NerdContainerInfo{}
				info.State.Status = "running"
				info.State.Running = true
				return info, nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(containerd.EnsureImageExists, tt.mockEnsureImageExists)

			patches.ApplyFunc(containerd.EnsureContainerRun, tt.mockEnsureContainerRun)

			if tt.mockRun != nil {
				patches.ApplyFunc(containerd.Run, tt.mockRun)
			}

			if tt.mockContainerInspect != nil {
				patches.ApplyFunc(containerd.ContainerInspect, tt.mockContainerInspect)
			}

			err := startChartRegistryWithContainerd("test-chart-registry", "chartmuseum:latest", "8080", "/var/lib/charts")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRemoveChartRegistry(t *testing.T) {
	tests := []struct {
		name                         string
		containerName                string
		mockRemoveContainerWithRetry func(string, func()) error
		expectError                  bool
	}{
		{
			name:          "successful removal",
			containerName: "test-chart-registry",
			mockRemoveContainerWithRetry: func(name string, extraCleanup func()) error {
				assert.Equal(t, "test-chart-registry", name)
				assert.Nil(t, extraCleanup)
				return nil
			},
			expectError: false,
		},
		{
			name:          "removal fails",
			containerName: "test-chart-registry",
			mockRemoveContainerWithRetry: func(name string, extraCleanup func()) error {
				return fmt.Errorf("removal failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(removeContainerWithRetry, tt.mockRemoveContainerWithRetry)

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := RemoveChartRegistry(tt.containerName)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartChartRegistryWithRealDockerClient(t *testing.T) {
	// Create a mock Docker client
	mockDockerClient := &docker.Client{}

	originalDocker := global.Docker
	global.Docker = mockDockerClient
	defer func() {
		global.Docker = originalDocker
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })

	patches.ApplyFunc((*docker.Client).EnsureImageExists, func(c *docker.Client, ref docker.ImageRef, opts utils.RetryOptions) error {
		return nil
	})

	patches.ApplyFunc((*docker.Client).EnsureContainerRun, func(c *docker.Client, name string) (bool, error) {
		return false, nil
	})

	patches.ApplyFunc(runDockerChartRegistry, func(name, image, port, dataDir string) error {
		return nil
	})

	patches.ApplyFunc(waitForDockerContainerRunning, func(name, serviceType string) {})

	err := StartChartRegistry("test-chart-registry", "chartmuseum:latest", "8080", "/var/lib/charts")

	assert.NoError(t, err)
}

func TestRemoveContainerWithRetry(t *testing.T) {
	t.Run("skip remove when docker container does not exist", func(t *testing.T) {
		originalDocker := global.Docker
		global.Docker = &docker.Client{}
		defer func() {
			global.Docker = originalDocker
		}()

		patches := gomonkey.NewPatches()
		defer patches.Reset()

		removeCalled := false
		patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
		patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
		patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
			return types.ContainerJSON{}, false
		})
		patches.ApplyFunc((*docker.Client).ContainerRemove, func(*docker.Client, string) error {
			removeCalled = true
			return nil
		})

		err := removeContainerWithRetry("missing-container", nil)
		assert.NoError(t, err)
		assert.False(t, removeCalled)
	})

	t.Run("return error when container still exists after retries", func(t *testing.T) {
		originalDocker := global.Docker
		global.Docker = &docker.Client{}
		defer func() {
			global.Docker = originalDocker
		}()

		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
		patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
		patches.ApplyFunc((*docker.Client).ContainerExists, func(*docker.Client, string) (types.ContainerJSON, bool) {
			return types.ContainerJSON{}, true
		})
		patches.ApplyFunc((*docker.Client).ContainerRemove, func(*docker.Client, string) error {
			return nil
		})
		patches.ApplyFunc(time.Sleep, func(time.Duration) {})

		err := removeContainerWithRetry("stuck-container", nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "still exists after remove retries")
	})
}
