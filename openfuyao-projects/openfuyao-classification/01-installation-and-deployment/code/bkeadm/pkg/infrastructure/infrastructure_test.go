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

package infrastructure

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	containerd_v2 "github.com/containerd/containerd/v2/client"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	cond "gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/dockerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testIPv4SegmentA        = 192
	testIPv4SegmentB        = 168
	testIPv4SegmentC        = 1
	testIPv4SegmentD        = 100
	testKubernetesPortValue = 6443
)

var testHostIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
).String()

var testKubernetesPort = fmt.Sprintf("%d", testKubernetesPortValue)

func TestRuntimeConfigStruct(t *testing.T) {
	// Test that the RuntimeConfig struct has the expected fields
	cfg := &RuntimeConfig{}

	cfg.Runtime = "docker"
	cfg.RuntimeStorage = "/var/lib/docker"
	cfg.Domain = "registry.example.com"
	cfg.ContainerdFile = "/path/to/containerd.tar.gz"
	cfg.CniPluginFile = "/path/to/cni.tar.gz"
	cfg.DockerdFile = "/path/to/dockerd.tar.gz"
	cfg.HostIP = testHostIP
	cfg.CAFile = "/path/to/ca.crt"

	assert.Equal(t, "docker", cfg.Runtime)
	assert.Equal(t, "/var/lib/docker", cfg.RuntimeStorage)
	assert.Equal(t, "registry.example.com", cfg.Domain)
	assert.Equal(t, "/path/to/containerd.tar.gz", cfg.ContainerdFile)
	assert.Equal(t, "/path/to/cni.tar.gz", cfg.CniPluginFile)
	assert.Equal(t, "/path/to/dockerd.tar.gz", cfg.DockerdFile)
	assert.Equal(t, testHostIP, cfg.HostIP)
	assert.Equal(t, "/path/to/ca.crt", cfg.CAFile)
}

func TestIsDocker(t *testing.T) {
	tests := []struct {
		name                string
		mockNewDockerClient func() (docker.DockerClient, error)
		mockGetClient       func() *client.Client
		mockPing            func(*client.Client, context.Context) (types.Ping, error)
		globalDocker        docker.DockerClient
		expectedResult      bool
	}{
		{
			name: "docker client is ready",
			mockNewDockerClient: func() (docker.DockerClient, error) {
				return &docker.Client{Client: &client.Client{}}, nil
			},
			mockGetClient: func() *client.Client {
				return &client.Client{}
			},
			mockPing: func(c *client.Client, ctx context.Context) (types.Ping, error) {
				return types.Ping{}, nil
			},
			globalDocker:   nil,
			expectedResult: true,
		},
		{
			name: "global docker client already exists and is ready",
			mockGetClient: func() *client.Client {
				return &client.Client{}
			},
			mockPing: func(c *client.Client, ctx context.Context) (types.Ping, error) {
				return types.Ping{}, nil
			},
			globalDocker:   &docker.Client{Client: &client.Client{}},
			expectedResult: true,
		},
		{
			name: "docker client ping fails",
			mockNewDockerClient: func() (docker.DockerClient, error) {
				return &docker.Client{Client: &client.Client{}}, nil
			},
			mockGetClient: func() *client.Client {
				return &client.Client{}
			},
			mockPing: func(c *client.Client, ctx context.Context) (types.Ping, error) {
				return types.Ping{}, fmt.Errorf("ping failed")
			},
			globalDocker:   nil,
			expectedResult: false,
		},
		{
			name: "docker client creation fails",
			mockNewDockerClient: func() (docker.DockerClient, error) {
				return nil, fmt.Errorf("client creation failed")
			},
			globalDocker:   nil,
			expectedResult: false,
		},
		{
			name: "global docker is nil and new client creation returns nil",
			mockNewDockerClient: func() (docker.DockerClient, error) {
				return nil, nil
			},
			globalDocker:   nil,
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up global Docker client
			originalDocker := global.Docker
			global.Docker = tt.globalDocker
			defer func() {
				global.Docker = originalDocker
			}()

			// Apply patches
			if tt.globalDocker == nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(docker.NewDockerClient, tt.mockNewDockerClient)
			}

			if tt.mockGetClient != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*docker.Client).GetClient, tt.mockGetClient)
			}

			if tt.mockPing != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*client.Client).Ping, tt.mockPing)
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(log.Info, func(v ...interface{}) {})

			result := IsDocker()

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestIsContainerd(t *testing.T) {
	tests := []struct {
		name                    string
		mockNewContainerdClient func() (containerd.ContainerdClient, error)
		mockGetClient           func() *containerd_v2.Client
		mockIsServing           func(*containerd_v2.Client, context.Context) (bool, error)
		mockExists              func(string) bool
		globalContainerd        containerd.ContainerdClient
		expectedResult          bool
	}{
		{
			name: "containerd client is serving and nerdctl exists",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return &containerd.Client{}, nil
			},
			mockGetClient: func() *containerd_v2.Client {
				return &containerd_v2.Client{}
			},
			mockIsServing: func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
				return true, nil
			},
			mockExists: func(path string) bool {
				return path == utils.NerdCtl
			},
			globalContainerd: nil,
			expectedResult:   true,
		},
		{
			name: "global containerd client already exists and is serving",
			mockGetClient: func() *containerd_v2.Client {
				return &containerd_v2.Client{}
			},
			mockIsServing: func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
				return true, nil
			},
			mockExists: func(path string) bool {
				return path == utils.NerdCtl
			},
			globalContainerd: &containerd.Client{},
			expectedResult:   true,
		},
		{
			name: "containerd client is serving but nerdctl doesn't exist",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return &containerd.Client{}, nil
			},
			mockGetClient: func() *containerd_v2.Client {
				return &containerd_v2.Client{}
			},
			mockIsServing: func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
				return true, nil
			},
			mockExists: func(path string) bool {
				return path != utils.NerdCtl
			},
			globalContainerd: nil,
			expectedResult:   false,
		},
		{
			name: "containerd client is not serving",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return &containerd.Client{}, nil
			},
			mockGetClient: func() *containerd_v2.Client {
				return &containerd_v2.Client{}
			},
			mockIsServing: func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
				return false, nil
			},
			globalContainerd: nil,
			expectedResult:   false,
		},
		{
			name: "containerd client serving returns error",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return &containerd.Client{}, nil
			},
			mockGetClient: func() *containerd_v2.Client {
				return &containerd_v2.Client{}
			},
			mockIsServing: func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
				return false, fmt.Errorf("serving error")
			},
			globalContainerd: nil,
			expectedResult:   false,
		},
		{
			name: "containerd client creation fails",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return nil, fmt.Errorf("client creation failed")
			},
			globalContainerd: nil,
			expectedResult:   false,
		},
		{
			name: "global containerd is nil and new client creation returns nil",
			mockNewContainerdClient: func() (containerd.ContainerdClient, error) {
				return nil, nil
			},
			globalContainerd: nil,
			expectedResult:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up global Containerd client
			originalContainerd := global.Containerd
			global.Containerd = tt.globalContainerd
			defer func() {
				global.Containerd = originalContainerd
			}()

			// Apply patches
			if tt.globalContainerd == nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(containerd.NewContainedClient, tt.mockNewContainerdClient)
			}

			if tt.mockGetClient != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*containerd.Client).GetClient, tt.mockGetClient)
			}

			if tt.mockIsServing != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*containerd_v2.Client).IsServing, tt.mockIsServing)
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)

			patches.ApplyFunc(log.Debug, func(v ...interface{}) {})

			result := IsContainerd()

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestRuntimeInstall(t *testing.T) {
	tests := []struct {
		name                   string
		cfg                    RuntimeConfig
		mockIsDocker           func() bool
		mockIsContainerd       func() bool
		mockEnsureDockerServer func(string, string, string, string) error
		mockInstall            func(string, string, string, string) error
		mockCniPluginInstall   func(string) error
		expectError            bool
	}{
		{
			name:             "runtime already installed",
			cfg:              RuntimeConfig{Runtime: "docker"},
			mockIsDocker:     func() bool { return true },
			mockIsContainerd: func() bool { return false },
			expectError:      false,
		},
		{
			name:             "both docker and containerd already installed",
			cfg:              RuntimeConfig{Runtime: "docker"},
			mockIsDocker:     func() bool { return true },
			mockIsContainerd: func() bool { return true },
			expectError:      false,
		},
		{
			name: "install docker runtime",
			cfg: RuntimeConfig{
				Runtime:        "docker",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/docker",
				DockerdFile:    "/path/to/dockerd.tar.gz",
				HostIP:         testHostIP,
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockEnsureDockerServer: func(domain, storage, dockerFile, hostIP string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "install containerd runtime",
			cfg: RuntimeConfig{
				Runtime:        "containerd",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/containerd",
				ContainerdFile: "/path/to/containerd.tar.gz",
				CAFile:         "/path/to/ca.crt",
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockInstall: func(domain, storage, containerdFile, caFile string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "install containerd with cni plugin",
			cfg: RuntimeConfig{
				Runtime:        "containerd",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/containerd",
				ContainerdFile: "/path/to/containerd.tar.gz",
				CAFile:         "/path/to/ca.crt",
				CniPluginFile:  "/path/to/cni.tar.gz",
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockInstall: func(domain, storage, containerdFile, caFile string) error {
				return nil
			},
			mockCniPluginInstall: func(cniPluginFile string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "docker installation fails",
			cfg: RuntimeConfig{
				Runtime:        "docker",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/docker",
				DockerdFile:    "/path/to/dockerd.tar.gz",
				HostIP:         testHostIP,
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockEnsureDockerServer: func(domain, storage, dockerFile, hostIP string) error {
				return fmt.Errorf("docker install failed")
			},
			expectError: true,
		},
		{
			name: "containerd installation fails",
			cfg: RuntimeConfig{
				Runtime:        "containerd",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/containerd",
				ContainerdFile: "/path/to/containerd.tar.gz",
				CAFile:         "/path/to/ca.crt",
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockInstall: func(domain, storage, containerdFile, caFile string) error {
				return fmt.Errorf("containerd install failed")
			},
			expectError: true,
		},
		{
			name: "cni plugin installation fails",
			cfg: RuntimeConfig{
				Runtime:        "containerd",
				Domain:         "registry.example.com",
				RuntimeStorage: "/var/lib/containerd",
				ContainerdFile: "/path/to/containerd.tar.gz",
				CAFile:         "/path/to/ca.crt",
				CniPluginFile:  "/path/to/cni.tar.gz",
			},
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			mockInstall: func(domain, storage, containerdFile, caFile string) error {
				return nil
			},
			mockCniPluginInstall: func(cniPluginFile string) error {
				return fmt.Errorf("cni install failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(IsContainerd, tt.mockIsContainerd)

			if !tt.mockIsDocker() && !tt.mockIsContainerd() {
				if tt.cfg.Runtime == "docker" {
					patches.ApplyFunc(dockerd.EnsureDockerServer, tt.mockEnsureDockerServer)
				} else if tt.cfg.Runtime == "containerd" {
					patches.ApplyFunc(cond.Install, tt.mockInstall)

					if tt.cfg.CniPluginFile != "" {
						patches.ApplyFunc(cond.CniPluginInstall, tt.mockCniPluginInstall)
					}
				}
			}

			err := RuntimeInstall(tt.cfg)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartLocalKubernetes(t *testing.T) {
	cfg := k3s.Config{
		HostIP:         testHostIP,
		KubernetesPort: testKubernetesPort,
	}
	localImage := "test-image:latest"

	tests := []struct {
		name                       string
		mockIsDocker               func() bool
		mockIsContainerd           func() bool
		mockStartK3sWithDocker     func(k3s.Config, string) error
		mockEnsureDirExists        func(string) error
		mockStartK3sWithContainerd func(k3s.Config, string) error
		expectError                bool
	}{
		{
			name:             "start with docker",
			mockIsDocker:     func() bool { return true },
			mockIsContainerd: func() bool { return false },
			mockStartK3sWithDocker: func(c k3s.Config, image string) error {
				assert.Equal(t, cfg, c)
				assert.Equal(t, localImage, image)
				return nil
			},
			expectError: false,
		},
		{
			name:             "start with containerd",
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return true },
			mockEnsureDirExists: func(dir string) error {
				assert.Equal(t, utils.DefaultExtendManifestsDir, dir)
				return nil
			},
			mockStartK3sWithContainerd: func(c k3s.Config, image string) error {
				assert.Equal(t, cfg, c)
				assert.Equal(t, localImage, image)
				return nil
			},
			expectError: false,
		},
		{
			name:             "start with both docker and containerd",
			mockIsDocker:     func() bool { return true },
			mockIsContainerd: func() bool { return true },
			mockStartK3sWithDocker: func(c k3s.Config, image string) error {
				assert.Equal(t, cfg, c)
				assert.Equal(t, localImage, image)
				return nil
			},
			mockEnsureDirExists: func(dir string) error {
				assert.Equal(t, utils.DefaultExtendManifestsDir, dir)
				return nil
			},
			mockStartK3sWithContainerd: func(c k3s.Config, image string) error {
				assert.Equal(t, cfg, c)
				assert.Equal(t, localImage, image)
				return nil
			},
			expectError: false,
		},
		{
			name:             "start with docker fails",
			mockIsDocker:     func() bool { return true },
			mockIsContainerd: func() bool { return false },
			mockStartK3sWithDocker: func(c k3s.Config, image string) error {
				return fmt.Errorf("k3s docker failed")
			},
			expectError: true,
		},
		{
			name:             "ensure dir fails",
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return true },
			mockEnsureDirExists: func(dir string) error {
				return fmt.Errorf("ensure dir failed")
			},
			expectError: true,
		},
		{
			name:             "start with containerd fails",
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return true },
			mockEnsureDirExists: func(dir string) error {
				return nil
			},
			mockStartK3sWithContainerd: func(c k3s.Config, image string) error {
				return fmt.Errorf("k3s containerd failed")
			},
			expectError: true,
		},
		{
			name:             "neither docker nor containerd available",
			mockIsDocker:     func() bool { return false },
			mockIsContainerd: func() bool { return false },
			expectError:      false, // Should return nil without error if neither is available
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(IsDocker, tt.mockIsDocker)

			patches.ApplyFunc(IsContainerd, tt.mockIsContainerd)

			if tt.mockIsDocker() {
				patches.ApplyFunc(k3s.StartK3sWithDocker, tt.mockStartK3sWithDocker)
			}

			if tt.mockIsContainerd() {
				if tt.mockEnsureDirExists != nil {
					patches.ApplyFunc(k3s.EnsureDirExists, tt.mockEnsureDirExists)
				}

				patches.ApplyFunc(k3s.StartK3sWithContainerd, tt.mockStartK3sWithContainerd)
			}

			err := StartLocalKubernetes(cfg, localImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestIsDockerWithTimeout(t *testing.T) {
	// Test the timeout functionality in IsDocker
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(docker.NewDockerClient, func() (docker.DockerClient, error) {
		return &docker.Client{Client: &client.Client{}}, nil
	})

	patches.ApplyFunc((*docker.Client).GetClient, func(c *docker.Client) *client.Client {
		return &client.Client{}
	})

	var capturedCtx context.Context
	patches.ApplyFunc((*client.Client).Ping, func(c *client.Client, ctx context.Context) (types.Ping, error) {
		capturedCtx = ctx
		return types.Ping{}, nil
	})

	patches.ApplyFunc(log.Info, func(v ...interface{}) {})

	// Reset global Docker client
	originalDocker := global.Docker
	global.Docker = nil
	defer func() {
		global.Docker = originalDocker
	}()

	result := IsDocker()

	assert.True(t, result)

	// Verify that the context had a timeout
	assert.NotNil(t, capturedCtx)
	deadline, ok := capturedCtx.Deadline()
	assert.True(t, ok)

	// The deadline should be set based on DefaultTimeoutSeconds
	expectedDeadline := time.Now().Add(utils.DefaultTimeoutSeconds * time.Second)
	assert.WithinDuration(t, expectedDeadline, deadline, time.Second)
}

func TestIsContainerdWithTimeout(t *testing.T) {
	// Test the timeout functionality in IsContainerd
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(containerd.NewContainedClient, func() (containerd.ContainerdClient, error) {
		return &containerd.Client{}, nil
	})

	patches.ApplyFunc((*containerd.Client).GetClient, func(c *containerd.Client) *containerd_v2.Client {
		return &containerd_v2.Client{}
	})

	var capturedCtx context.Context
	patches.ApplyFunc((*containerd_v2.Client).IsServing, func(c *containerd_v2.Client, ctx context.Context) (bool, error) {
		capturedCtx = ctx
		return true, nil
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == utils.NerdCtl
	})

	patches.ApplyFunc(log.Debug, func(v ...interface{}) {})

	// Reset global Containerd client
	originalContainerd := global.Containerd
	global.Containerd = nil
	defer func() {
		global.Containerd = originalContainerd
	}()

	result := IsContainerd()

	assert.True(t, result)

	// Verify that the context had a timeout
	assert.NotNil(t, capturedCtx)
	deadline, ok := capturedCtx.Deadline()
	assert.True(t, ok)

	// The deadline should be set based on DefaultTimeoutSeconds
	expectedDeadline := time.Now().Add(utils.DefaultTimeoutSeconds * time.Second)
	assert.WithinDuration(t, expectedDeadline, deadline, time.Second)
}

func TestRuntimeInstallWithDockerRuntime(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime:        "docker",
		Domain:         "registry.example.com",
		RuntimeStorage: "/var/lib/docker",
		DockerdFile:    "/path/to/dockerd.tar.gz",
		HostIP:         testHostIP,
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	var capturedDomain, capturedStorage, capturedDockerFile, capturedHostIP string
	patches.ApplyFunc(dockerd.EnsureDockerServer, func(domain, storage, dockerFile, hostIP string) error {
		capturedDomain = domain
		capturedStorage = storage
		capturedDockerFile = dockerFile
		capturedHostIP = hostIP
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)
	assert.Equal(t, "registry.example.com", capturedDomain)
	assert.Equal(t, "/var/lib/docker", capturedStorage)
	assert.Equal(t, "/path/to/dockerd.tar.gz", capturedDockerFile)
	assert.Equal(t, testHostIP, capturedHostIP)
}

func TestRuntimeInstallWithContainerdRuntime(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime:        "containerd",
		Domain:         "registry.example.com",
		RuntimeStorage: "/var/lib/containerd",
		ContainerdFile: "/path/to/containerd.tar.gz",
		CAFile:         "/path/to/ca.crt",
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(cond.Install, func(domain, port, storage, containerdFile, caFile string) error {

		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)

}

func TestRuntimeInstallWithCniPlugin(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime:        "containerd",
		Domain:         "registry.example.com",
		RuntimeStorage: "/var/lib/containerd",
		ContainerdFile: "/path/to/containerd.tar.gz",
		CAFile:         "/path/to/ca.crt",
		CniPluginFile:  "/path/to/cni.tar.gz",
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(cond.Install, func(domain, storage, containerdFile, caFile string) error {
		return nil
	})

	var capturedCniPluginFile string
	patches.ApplyFunc(cond.CniPluginInstall, func(cniPluginFile string) error {
		capturedCniPluginFile = cniPluginFile
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)
	assert.Equal(t, "/path/to/cni.tar.gz", capturedCniPluginFile)
}

func TestStartLocalKubernetesWithBothAvailable(t *testing.T) {
	cfg := k3s.Config{
		HostIP:         testHostIP,
		KubernetesPort: testKubernetesPort,
	}
	localImage := "test-image:latest"

	// Test when both Docker and Containerd are available - both should be called
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return true })

	patches.ApplyFunc(IsContainerd, func() bool { return true })

	dockerCalled := false
	containerdCalled := false

	patches.ApplyFunc(k3s.StartK3sWithDocker, func(c k3s.Config, image string) error {
		dockerCalled = true
		assert.Equal(t, cfg, c)
		assert.Equal(t, localImage, image)
		return nil
	})

	patches.ApplyFunc(k3s.EnsureDirExists, func(dir string) error {
		return nil
	})

	patches.ApplyFunc(k3s.StartK3sWithContainerd, func(c k3s.Config, image string) error {
		containerdCalled = true
		assert.Equal(t, cfg, c)
		assert.Equal(t, localImage, image)
		return nil
	})

	err := StartLocalKubernetes(cfg, localImage)

	assert.NoError(t, err)
	assert.True(t, dockerCalled)
	assert.True(t, containerdCalled)
}

func TestStartLocalKubernetesWithNeitherAvailable(t *testing.T) {
	cfg := k3s.Config{
		HostIP:         testHostIP,
		KubernetesPort: testKubernetesPort,
	}
	localImage := "test-image:latest"

	// Test when neither Docker nor Containerd is available
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	// No other functions should be called in this case
	err := StartLocalKubernetes(cfg, localImage)

	assert.NoError(t, err)
}

func TestRuntimeConfigWithEmptyValues(t *testing.T) {
	// Test RuntimeConfig with empty values
	cfg := &RuntimeConfig{}

	assert.Equal(t, "", cfg.Runtime)
	assert.Equal(t, "", cfg.RuntimeStorage)
	assert.Equal(t, "", cfg.Domain)
	assert.Equal(t, "", cfg.ContainerdFile)
	assert.Equal(t, "", cfg.CniPluginFile)
	assert.Equal(t, "", cfg.DockerdFile)
	assert.Equal(t, "", cfg.HostIP)
	assert.Equal(t, "", cfg.CAFile)
}

func TestRuntimeInstallWithEmptyConfig(t *testing.T) {
	cfg := RuntimeConfig{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(dockerd.EnsureDockerServer, func(domain, storage, dockerFile, hostIP string) error {
		// All values should be empty strings
		assert.Equal(t, "", domain)
		assert.Equal(t, "", storage)
		assert.Equal(t, "", dockerFile)
		assert.Equal(t, "", hostIP)
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.Error(t, err)
}

func TestRuntimeInstallDockerRuntimeWithEmptyValues(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime: "docker",
		// Other fields are empty
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(dockerd.EnsureDockerServer, func(domain, storage, dockerFile, hostIP string) error {
		// All values should be empty strings
		assert.Equal(t, "", domain)
		assert.Equal(t, "", storage)
		assert.Equal(t, "", dockerFile)
		assert.Equal(t, "", hostIP)
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)
}

func TestRuntimeInstallContainerdRuntimeWithEmptyValues(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime: "containerd",
		// Other fields are empty
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(cond.Install, func(domain, storage, containerdFile, caFile string) error {
		// All values should be empty strings
		assert.Equal(t, "", domain)
		assert.Equal(t, "", storage)
		assert.Equal(t, "", containerdFile)
		assert.Equal(t, "", caFile)
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)
}

func TestRuntimeInstallContainerdWithCniPluginEmpty(t *testing.T) {
	cfg := RuntimeConfig{
		Runtime:       "containerd",
		CniPluginFile: "", // Empty CNI plugin file
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })

	patches.ApplyFunc(IsContainerd, func() bool { return false })

	patches.ApplyFunc(cond.Install, func(domain, storage, containerdFile, caFile string) error {
		return nil
	})

	// CniPluginInstall should not be called since CniPluginFile is empty
	patches.ApplyFunc(cond.CniPluginInstall, func(cniPluginFile string) error {
		t.Error("CniPluginInstall should not be called when CniPluginFile is empty")
		return nil
	})

	err := RuntimeInstall(cfg)

	assert.NoError(t, err)
}
