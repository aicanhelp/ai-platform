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
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testRegistryName    = "test-registry"
	testRegistryImage   = "registry:2"
	testRegistryPort    = "5000"
	testRegistryDataDir = "/var/lib/registry"
)

func TestStartImageRegistryWithContainerd(t *testing.T) {
	tests := []struct {
		name                             string
		mockGenerateConfig               func(string, string) error
		mockContainerdEnsureImageExists  func(string) error
		mockContainerdEnsureContainerRun func(string) (bool, error)
		expectError                      bool
	}{
		{
			name: "containerd container already running",
			mockGenerateConfig: func(certPath, port string) error {
				return nil
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
			name: "generate config fails",
			mockGenerateConfig: func(certPath, port string) error {
				return assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			if tt.mockGenerateConfig != nil {
				patches.ApplyFunc(generateConfig, tt.mockGenerateConfig)
			}
			if tt.mockContainerdEnsureImageExists != nil {
				patches.ApplyFunc(containerd.EnsureImageExists, tt.mockContainerdEnsureImageExists)
			}
			if tt.mockContainerdEnsureContainerRun != nil {
				patches.ApplyFunc(containerd.EnsureContainerRun, tt.mockContainerdEnsureContainerRun)
			}

			err := startImageRegistryWithContainerd(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRemoveImageRegistry(t *testing.T) {
	tests := []struct {
		name                string
		mockIsDocker        func() bool
		mockIsContainerd    func() bool
		mockRemoveContainer func(string) error
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
			mockRemoveContainer: func(name string) error {
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
			mockRemoveContainer: func(name string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)

			patches.ApplyFunc(removeContainerWithRetry, func(name string, extraCleanup func()) error {
				return tt.mockRemoveContainer(name)
			})

			err := RemoveImageRegistry(testRegistryName)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGenerateConfig(t *testing.T) {
	tests := []struct {
		name                          string
		certPath                      string
		port                          string
		mockSetRegistryConfig         func(string) error
		mockSetServerCertificate      func(string) error
		mockSetClientCertificate      func(string, string) error
		mockSetClientLocalCertificate func(string, string) error
		expectError                   bool
	}{
		{
			name:     "successful config generation",
			certPath: "/etc/docker/test-registry",
			port:     "5000",
			mockSetRegistryConfig: func(certPath string) error {
				return nil
			},
			mockSetServerCertificate: func(certPath string) error {
				return nil
			},
			mockSetClientCertificate: func(certPath, port string) error {
				return nil
			},
			mockSetClientLocalCertificate: func(certPath, port string) error {
				return nil
			},
			expectError: false,
		},
		{
			name:     "set registry config fails",
			certPath: "/etc/docker/test-registry",
			port:     "5000",
			mockSetRegistryConfig: func(certPath string) error {
				return assert.AnError
			},
			expectError: true,
		},
		{
			name:     "set server certificate fails",
			certPath: "/etc/docker/test-registry",
			port:     "5000",
			mockSetRegistryConfig: func(certPath string) error {
				return nil
			},
			mockSetServerCertificate: func(certPath string) error {
				return assert.AnError
			},
			expectError: true,
		},
		{
			name:     "set client certificate fails",
			certPath: "/etc/docker/test-registry",
			port:     "5000",
			mockSetRegistryConfig: func(certPath string) error {
				return nil
			},
			mockSetServerCertificate: func(certPath string) error {
				return nil
			},
			mockSetClientCertificate: func(certPath, port string) error {
				return assert.AnError
			},
			expectError: true,
		},
		{
			name:     "set client local certificate fails",
			certPath: "/etc/docker/test-registry",
			port:     "5000",
			mockSetRegistryConfig: func(certPath string) error {
				return nil
			},
			mockSetServerCertificate: func(certPath string) error {
				return nil
			},
			mockSetClientCertificate: func(certPath, port string) error {
				return nil
			},
			mockSetClientLocalCertificate: func(certPath, port string) error {
				return assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			if tt.mockSetRegistryConfig != nil {
				patches.ApplyFunc(SetRegistryConfig, tt.mockSetRegistryConfig)
			}
			if tt.mockSetServerCertificate != nil {
				patches.ApplyFunc(SetServerCertificate, tt.mockSetServerCertificate)
			}
			if tt.mockSetClientCertificate != nil {
				patches.ApplyFunc(SetClientCertificate, tt.mockSetClientCertificate)
			}
			if tt.mockSetClientLocalCertificate != nil {
				patches.ApplyFunc(SetClientLocalCertificate, tt.mockSetClientLocalCertificate)
			}

			err := generateConfig(tt.certPath, tt.port)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRunDockerImageRegistry(t *testing.T) {
	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	defer func() { global.Docker = originalDocker }()

	t.Run("successful run", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*docker.Client).Run, func(_ *docker.Client, _ *container.Config, _ *container.HostConfig,
			_ *network.NetworkingConfig, _ *specs.Platform, _ string) error {
			return nil
		})

		err := runDockerImageRegistry(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir, "/etc/certs")
		assert.NoError(t, err)
	})

	t.Run("docker run fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*docker.Client).Run, func(_ *docker.Client, _ *container.Config, _ *container.HostConfig,
			_ *network.NetworkingConfig, _ *specs.Platform, _ string) error {
			return fmt.Errorf("run failed")
		})

		err := runDockerImageRegistry(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir, "/etc/certs")
		assert.Error(t, err)
	})
}

func TestStartImageRegistry(t *testing.T) {
	tests := []struct {
		name                    string
		mockIsContainerd        func() bool
		mockIsDocker            func() bool
		mockStartWithContainerd func(string, string, string, string) error
		mockStartWithDocker     func(string, string, string, string) error
		expectError             bool
	}{
		{
			name:                    "containerd only",
			mockIsContainerd:        func() bool { return true },
			mockIsDocker:            func() bool { return false },
			mockStartWithContainerd: func(string, string, string, string) error { return nil },
			expectError:             false,
		},
		{
			name:                "docker available",
			mockIsContainerd:    func() bool { return false },
			mockIsDocker:        func() bool { return true },
			mockStartWithDocker: func(string, string, string, string) error { return nil },
			expectError:         false,
		},
		{
			name:             "both runtimes prefer docker",
			mockIsContainerd: func() bool { return true },
			mockIsDocker:     func() bool { return true },
			mockStartWithDocker: func(string, string, string, string) error { return nil },
			mockStartWithContainerd: func(string, string, string, string) error {
				t.Fatal("containerd path should not run when docker is available")
				return nil
			},
			expectError: false,
		},
		{
			name:             "no runtime",
			mockIsContainerd: func() bool { return false },
			mockIsDocker:     func() bool { return false },
			expectError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(infrastructure.IsContainerd, tt.mockIsContainerd)
			patches.ApplyFunc(infrastructure.IsDocker, tt.mockIsDocker)
			if tt.mockStartWithContainerd != nil {
				patches.ApplyFunc(startImageRegistryWithContainerd, tt.mockStartWithContainerd)
			}
			if tt.mockStartWithDocker != nil {
				patches.ApplyFunc(startImageRegistryWithDocker, tt.mockStartWithDocker)
			}

			err := StartImageRegistry(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartImageRegistryWithDocker(t *testing.T) {
	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	defer func() { global.Docker = originalDocker }()

	t.Run("deploy new container", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(generateConfig, func(string, string) error { return nil })
		patches.ApplyFunc((*docker.Client).EnsureImageExists, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
			return nil
		})
		patches.ApplyFunc((*docker.Client).EnsureContainerRun, func(_ *docker.Client, _ string) (bool, error) {
			return false, nil
		})
		patches.ApplyFunc(runDockerImageRegistry, func(string, string, string, string, string) error { return nil })
		patches.ApplyFunc(waitForDockerContainerRunning, func(string, string) {})

		err := startImageRegistryWithDocker(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
		assert.NoError(t, err)
	})

	t.Run("already running", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(generateConfig, func(string, string) error { return nil })
		patches.ApplyFunc((*docker.Client).EnsureImageExists, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
			return nil
		})
		patches.ApplyFunc((*docker.Client).EnsureContainerRun, func(_ *docker.Client, _ string) (bool, error) {
			return true, nil
		})

		err := startImageRegistryWithDocker(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
		assert.NoError(t, err)
	})

	t.Run("generate config fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(generateConfig, func(string, string) error { return assert.AnError })

		err := startImageRegistryWithDocker(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
		assert.Error(t, err)
	})
}

func TestStartImageRegistryWithContainerdDeploy(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(generateConfig, func(string, string) error { return nil })
	patches.ApplyFunc(containerd.EnsureImageExists, func(string) error { return nil })
	patches.ApplyFunc(containerd.EnsureContainerRun, func(string) (bool, error) { return false, nil })
	patches.ApplyFunc(containerd.Run, func([]string) error { return nil })
	patches.ApplyFunc(containerd.ContainerInspect, func(string) (containerd.NerdContainerInfo, error) {
		info := containerd.NerdContainerInfo{}
		info.State.Running = true
		return info, nil
	})

	err := startImageRegistryWithContainerd(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
	assert.NoError(t, err)
}

func TestStartImageRegistryWithContainerdRunFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(generateConfig, func(string, string) error { return nil })
	patches.ApplyFunc(containerd.EnsureImageExists, func(string) error { return nil })
	patches.ApplyFunc(containerd.EnsureContainerRun, func(string) (bool, error) { return false, nil })
	patches.ApplyFunc(containerd.Run, func([]string) error { return fmt.Errorf("run failed") })

	err := startImageRegistryWithContainerd(testRegistryName, testRegistryImage, testRegistryPort, testRegistryDataDir)
	assert.Error(t, err)
}

func TestWaitForDockerContainerRunningImage(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	defer func() { global.Docker = originalDocker }()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*docker.Client).GetClient, func(*docker.Client) *client.Client {
		return &client.Client{}
	})
	patches.ApplyFunc((*client.Client).ContainerInspect, func(*client.Client, context.Context, string) (container.InspectResponse, error) {
		return container.InspectResponse{
			ContainerJSONBase: &types.ContainerJSONBase{State: &types.ContainerState{Running: true}},
		}, nil
	})

	waitForDockerContainerRunning(testRegistryName, "container")
}
