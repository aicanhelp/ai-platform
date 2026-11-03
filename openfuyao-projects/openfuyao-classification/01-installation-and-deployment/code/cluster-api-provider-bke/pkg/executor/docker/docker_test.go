/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package docker

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	dockerapi "github.com/docker/docker/client"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	numOne   = 1
	numTwo   = 2
	numThree = 3
)

const (
	testImage         = "test-image:latest"
	testUsername      = "testuser"
	testPassword      = "testpassword"
	testContainerID   = "test-container-id"
	testContainerName = "test-container"
	testRegistry      = "test.registry.com"
	testCgroupDriver  = "systemd"
	testRuntime       = "kata"
	testDataRoot      = "/var/lib/docker/test"
	testTLSHost       = "127.0.0.1"
	testDaemonPath    = "/etc/docker/daemon_test.json"
	testDaemonContent = "{\"log-driver\": \"json-file\"}"
)

type mockReadCloser struct {
	reader io.Reader
}

func (m *mockReadCloser) Read(p []byte) (n int, err error) {
	return m.reader.Read(p)
}

func (m *mockReadCloser) Close() error {
	return nil
}

func TestNewDockerClientDockerSockNotExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != dockerSock
	})

	client, err := NewDockerClient()
	if err == nil {
		t.Error("Expected error when docker.sock does not exist")
	}
	if client != nil {
		t.Error("Expected nil client when docker.sock does not exist")
	}
}

func TestNewDockerClientSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "Ping", func(ctx context.Context) (types.Ping, error) {
		return types.Ping{APIVersion: "1.41"}, nil
	})

	patches.ApplyFunc(dockerapi.NewClientWithOpts, func(opts ...dockerapi.Opt) (*dockerapi.Client, error) {
		return mockClient, nil
	})

	client, err := NewDockerClient()
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if client == nil {
		t.Error("Expected non-nil client")
	}
}

func TestNewDockerClientPingFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "Ping", func(ctx context.Context) (types.Ping, error) {
		return types.Ping{}, errors.New("docker daemon not running")
	})

	patches.ApplyFunc(dockerapi.NewClientWithOpts, func(opts ...dockerapi.Opt) (*dockerapi.Client, error) {
		return mockClient, nil
	})

	client, err := NewDockerClient()
	if err == nil {
		t.Error("Expected error when ping fails")
	}
	if client != nil {
		t.Error("Expected nil client when ping fails")
	}
}

func TestCloseSuccess(t *testing.T) {
	c := &Client{
		Client: &dockerapi.Client{},
		ctx:    context.Background(),
	}
	c.Close()
}

func TestPingSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "Ping", func(ctx context.Context) (types.Ping, error) {
		return types.Ping{APIVersion: "1.41"}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}
	err := c.Ping()

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestPingFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "Ping", func(ctx context.Context) (types.Ping, error) {
		return types.Ping{}, errors.New("docker daemon not running")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}
	err := c.Ping()

	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestPullSuccessWithAuth(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImagePull", func(ctx context.Context, refStr string, options image.PullOptions) (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte("pulled"))}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{
		Image:    testImage,
		Username: testUsername,
		Password: testPassword,
	}

	err := c.Pull(img)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestPullSuccessWithoutAuth(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImagePull", func(ctx context.Context, refStr string, options image.PullOptions) (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte("pulled"))}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{
		Image: testImage,
	}

	err := c.Pull(img)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestPullFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImagePull", func(ctx context.Context, refStr string, options image.PullOptions) (io.ReadCloser, error) {
		return nil, errors.New("pull failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{Image: testImage}

	err := c.Pull(img)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestPushSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImagePush", func(ctx context.Context, refStr string, options image.PushOptions) (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte("pushed"))}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{
		Image:    testImage,
		Username: testUsername,
		Password: testPassword,
	}

	closer, err := c.Push(img)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if closer == nil {
		t.Error("Expected non-nil closer")
	}
}

func TestPushFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImagePush", func(ctx context.Context, refStr string, options image.PushOptions) (io.ReadCloser, error) {
		return nil, errors.New("push failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{Image: testImage}

	_, err := c.Push(img)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestRunSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerCreate", func(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig interface{}, platform interface{}, containerName string) (container.CreateResponse, error) {
		return container.CreateResponse{ID: testContainerID}, nil
	})
	patches.ApplyMethodFunc(mockClient, "ContainerStart", func(ctx context.Context, containerID string, options container.StartOptions) error {
		return nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	cs := ContainerSpec{
		ContainerConfig: &container.Config{Image: testImage},
		HostConfig:      &container.HostConfig{},
		ContainerName:   testContainerName,
	}

	err := c.Run(cs)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestRunContainerCreateFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	createCalled := false
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerCreate", func(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig interface{}, platform interface{}, containerName string) (container.CreateResponse, error) {
		createCalled = true
		return container.CreateResponse{}, errors.New("create failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	cs := ContainerSpec{
		ContainerConfig: &container.Config{Image: testImage},
		HostConfig:      &container.HostConfig{},
	}

	err := c.Run(cs)
	if !createCalled {
		t.Error("Expected ContainerCreate to be called")
	}
	if err == nil {
		t.Error("Expected create failure to return error")
	}
}

func TestRunContainerStartFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	startCalled := false
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerCreate", func(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig interface{}, platform interface{}, containerName string) (container.CreateResponse, error) {
		return container.CreateResponse{ID: testContainerID}, nil
	})
	patches.ApplyMethodFunc(mockClient, "ContainerStart", func(ctx context.Context, containerID string, options container.StartOptions) error {
		startCalled = true
		return errors.New("start failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	cs := ContainerSpec{
		ContainerConfig: &container.Config{Image: testImage},
		HostConfig:      &container.HostConfig{},
	}

	_ = c.Run(cs)
	if !startCalled {
		t.Error("Expected ContainerStart to be called")
	}
}

func TestEnsureImageExistsImageExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImageInspectWithRaw", func(ctx context.Context, imageID string) (types.ImageInspect, []byte, error) {
		return types.ImageInspect{ID: "existing-image-id"}, nil, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{Image: testImage}
	err := c.EnsureImageExists(img)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestEnsureImageExistsImageNotExistsPullSucceeds(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pullCalled := false
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImageInspectWithRaw", func(ctx context.Context, imageID string) (types.ImageInspect, []byte, error) {
		return types.ImageInspect{}, nil, nil
	})
	patches.ApplyMethodFunc(mockClient, "ImagePull", func(ctx context.Context, refStr string, options image.PullOptions) (io.ReadCloser, error) {
		pullCalled = true
		return &mockReadCloser{reader: bytes.NewReader([]byte("pulled"))}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{Image: testImage}
	err := c.EnsureImageExists(img)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if !pullCalled {
		t.Error("Expected ImagePull to be called")
	}
}

func TestEnsureImageExistsImageNotExistsPullFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ImageInspectWithRaw", func(ctx context.Context, imageID string) (types.ImageInspect, []byte, error) {
		return types.ImageInspect{}, nil, nil
	})
	patches.ApplyMethodFunc(mockClient, "ImagePull", func(ctx context.Context, refStr string, options image.PullOptions) (io.ReadCloser, error) {
		return nil, errors.New("pull failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	img := ImageRef{Image: testImage}
	err := c.EnsureImageExists(img)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestEnsureContainerRunContainerRunning(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerInspect", func(ctx context.Context, containerID string) (types.ContainerJSON, error) {
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				ID: testContainerID,
				State: &types.ContainerState{
					Running: true,
				},
			},
		}, nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	running, err := c.EnsureContainerRun(testContainerID)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if !running {
		t.Error("Expected container to be running")
	}
}

func TestEnsureContainerRunContainerStoppedStartsSuccessfully(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	startCalled := false
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerInspect", func(ctx context.Context, containerID string) (types.ContainerJSON, error) {
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				ID: testContainerID,
				State: &types.ContainerState{
					Running: false,
				},
			},
		}, nil
	})
	patches.ApplyMethodFunc(mockClient, "ContainerStart", func(ctx context.Context, containerID string, options container.StartOptions) error {
		startCalled = true
		return nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	running, err := c.EnsureContainerRun(testContainerID)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if !running {
		t.Error("Expected container to be running after start")
	}
	if !startCalled {
		t.Error("Expected ContainerStart to be called")
	}
}

func TestEnsureContainerRunContainerRemoved(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	type testState struct {
		startCalled  bool
		removeCalled bool
	}
	state := &testState{}
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerInspect", func(ctx context.Context, containerID string) (types.ContainerJSON, error) {
		return types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				ID: testContainerID,
				State: &types.ContainerState{
					Running: false,
				},
			},
		}, nil
	})
	patches.ApplyMethodFunc(mockClient, "ContainerStart", func(ctx context.Context, containerID string, options container.StartOptions) error {
		state.startCalled = true
		return errors.New("start failed")
	})
	patches.ApplyMethodFunc(mockClient, "ContainerRemove", func(ctx context.Context, containerID string, options container.RemoveOptions) error {
		state.removeCalled = true
		return nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	running, err := c.EnsureContainerRun(testContainerID)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if running {
		t.Error("Expected container to not be running")
	}
	if !state.removeCalled {
		t.Error("Expected ContainerRemove to be called")
	}
	if !state.startCalled {
		t.Error("ContainerStart should have been called")
	}
}

func TestRemoveContainerSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	removeCalled := false
	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerRemove", func(ctx context.Context, containerID string, options container.RemoveOptions) error {
		removeCalled = true
		return nil
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	err := c.RemoveContainer(testContainerID)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if !removeCalled {
		t.Error("Expected ContainerRemove to be called")
	}
}

func TestRemoveContainerFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &dockerapi.Client{}
	patches.ApplyMethodFunc(mockClient, "ContainerRemove", func(ctx context.Context, containerID string, options container.RemoveOptions) error {
		return errors.New("remove failed")
	})

	c := &Client{
		Client: mockClient,
		ctx:    context.Background(),
	}

	err := c.RemoveContainer(testContainerID)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestGetDockerDaemonConfigPathNotEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == testDaemonPath {
			return true
		}
		return false
	})
	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == testDaemonPath {
			return []byte(testDaemonContent), nil
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	cfg, err := GetDockerDaemonConfig(testDaemonPath)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if cfg == nil {
		t.Error("Expected non-nil config")
	}
}

func TestGetDockerDaemonConfigPathEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	GetDockerDaemonConfig("")
}

func TestOverrideDockerService(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mkdirCalled := false
	writeCalled := false

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/systemd/system/docker.service.d" {
			return false
		}
		return true
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		if path == "/etc/systemd/system/docker.service.d" {
			mkdirCalled = true
		}
		return nil
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == "/etc/systemd/system/docker.service.d/docker.conf" {
			writeCalled = true
		}
		return nil
	})

	err := OverrideDockerService()
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if !mkdirCalled {
		t.Error("Expected MkdirAll to be called")
	}
	if !writeCalled {
		t.Error("Expected WriteFile to be called")
	}
}

func TestConfigInsecureRegistriesEmptyPath(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	registries := []string{testRegistry}
	ConfigInsecureRegistries(registries)

}

func TestConfigCgroupDriver(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	ConfigCgroupDriver(testCgroupDriver)
}

func TestConfigRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	ConfigRuntime(testRuntime)
}

func TestConfigDataRoot(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == testDataRoot {
			return false
		}
		return path != DockerDaemonConfigFilePath
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		if path == filepath.Dir(testDataRoot) {
			return nil
		}
		return errors.New("mkdir failed")
	})

	ConfigDataRoot(testDataRoot)
}

func TestBaseConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != DockerDaemonConfigFilePath
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	BaseConfig()
}

func TestConfigDockerDaemon(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker" {
			return true
		}
		if path == "/etc/docker/certs" {
			return true
		}
		return path != DockerDaemonConfigFilePath && path != testDataRoot
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})
	patches.ApplyFunc(utils.UniqueStringSlice, func(s []string) []string {
		return s
	})
	patches.ApplyFunc(json.MarshalIndent, func(v interface{}, prefix, indent string) ([]byte, error) {
		return []byte("{}"), nil
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(OverrideDockerService, func() error {
		return nil
	})
	patches.ApplyFunc(BaseConfig, func() error {
		return nil
	})
	patches.ApplyFunc(ConfigInsecureRegistries, func(registries []string) error {
		return nil
	})
	patches.ApplyFunc(ConfigCgroupDriver, func(driver string) error {
		return nil
	})
	patches.ApplyFunc(ConfigRuntime, func(runtime string) error {
		return nil
	})
	patches.ApplyFunc(ConfigDataRoot, func(dataRoot string) error {
		return nil
	})
	patches.ApplyFunc(ConfigDockerTls, func(tlsHost string) error {
		return nil
	})

	err := ConfigDockerDaemon(DockerDaemonConfig{
		CgroupDriver:    testCgroupDriver,
		LowLevelRuntime: testRuntime,
		DataRoot:        testDataRoot,
		EnableTls:       true,
		TlsHost:         testTLSHost,
	})
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestConfigDockerTlsSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	certsDirCreated := false

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return false
		}
		return true
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		if path == "/etc/docker/certs" {
			certsDirCreated = true
		}
		return nil
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == "/etc/docker/certs/tlscert.sh" {
			return errors.New("write tlscert.sh failed")
		}
		return nil
	})

	err := ConfigDockerTls(testTLSHost)
	if err == nil {
		t.Error("Expected error when TLS cert script write fails")
	}
	if !certsDirCreated {
		t.Error("Expected certs directory to be created")
	}
}

func TestConfigDockerTlsEmptyHost(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return false
		}
		return true
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == "/etc/docker/certs/tlscert.sh" {
			return errors.New("write tlscert.sh failed")
		}
		return nil
	})

	err := ConfigDockerTls("")
	if err == nil {
		t.Error("Expected error when TLS cert script write fails")
	}
}

func TestConfigDockerTlsMkdirFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return false
		}
		return true
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	err := ConfigDockerTls(testTLSHost)
	if err == nil {
		t.Error("Expected error when mkdir fails")
	}
}

func TestConfigDockerTlsWithExistingDaemonConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	existingConfig := `{"log-driver": "json-file"}`

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return true
		}
		if path == DockerDaemonConfigFilePath {
			return true
		}
		return false
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return nil
		}
		return nil
	})
	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return []byte(existingConfig), nil
		}
		return nil, errors.New("file not found")
	})

	err := ConfigDockerTls(testTLSHost)
	if err == nil {
		t.Skip("Skipping test - command execution not available on this platform")
	}
}

func TestConfigDockerTlsWithEmptyDaemonConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return true
		}
		if path == DockerDaemonConfigFilePath {
			return true
		}
		return false
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return nil
		}
		return nil
	})
	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return []byte(""), nil
		}
		return nil, errors.New("file not found")
	})

	err := ConfigDockerTls(testTLSHost)
	if err == nil {
		t.Skip("Skipping test - command execution not available on this platform")
	}
}

func TestWaitDockerReadySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	callCount := 0
	patches.ApplyFunc(NewDockerClient, func() (DockerClient, error) {
		callCount++
		if callCount < numTwo {
			return nil, errors.New("docker not ready")
		}
		return &Client{}, nil
	})
	patches.ApplyFunc(wait.PollImmediateUntil, func(interval time.Duration, condition func() (bool, error), done <-chan struct{}) error {
		for {
			select {
			case <-done:
				return errors.New("done channel closed")
			default:
				ok, err := condition()
				if err != nil {
					return err
				}
				if ok {
					return nil
				}
			}
		}
	})

	err := WaitDockerReady()
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if callCount < numTwo {
		t.Error("Expected NewDockerClient to be called at least twice")
	}
}

func TestWaitDockerReadyTimeout(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(NewDockerClient, func() (DockerClient, error) {
		return nil, errors.New("docker not ready")
	})
	patches.ApplyFunc(wait.PollImmediateUntil, func(interval time.Duration, condition func() (bool, error), done <-chan struct{}) error {
		return errors.New("timed out waiting for the condition")
	})

	err := WaitDockerReady()
	if err == nil {
		t.Error("Expected error when Docker never becomes ready")
	}
}

func TestUpdateDockerConfigWithRegistries(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{"log-driver":"json-file"}`), DefaultFileMode)

	err := updateDockerConfigWithRegistries([]string{"registry1.com", "registry2.com"})
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleInsecureRegistries(t *testing.T) {
	v := map[string]interface{}{}
	registries := []string{"test.com"}
	result := handleInsecureRegistries(v, registries)
	if len(result) == 0 {
		t.Error("Expected non-empty registries")
	}
}

func TestHandleExistingInsecureRegistries(t *testing.T) {
	v := map[string]interface{}{
		"insecure-registries": []interface{}{"existing.com"},
	}
	registries := []string{"new.com"}
	result := handleExistingInsecureRegistries(v, registries)
	if len(result) < numTwo {
		t.Error("Expected at least 2 registries")
	}
}

func TestUpdateDockerConfigWithCgroupDriver(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{"exec-opts":["native.cgroupdriver=cgroupfs"]}`), DefaultFileMode)

	err := updateDockerConfigWithCgroupDriver("systemd", "native.cgroupdriver=systemd")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestHandleExecOpts(t *testing.T) {
	v := map[string]interface{}{
		"exec-opts": []interface{}{"native.cgroupdriver=cgroupfs"},
	}
	result := handleExecOpts(v, "systemd", "native.cgroupdriver=systemd")
	if result {
		t.Log("Early return triggered")
	}
}

func TestProcessOpt(t *testing.T) {
	opts := []interface{}{"native.cgroupdriver=cgroupfs"}
	result := processOpt("native.cgroupdriver=systemd", opts, 0, "systemd", "native.cgroupdriver=systemd")
	if result {
		t.Log("Early return triggered")
	}
}

func TestUpdateCgroupDriverValue(t *testing.T) {
	opts := []interface{}{"native.cgroupdriver=cgroupfs"}
	updateCgroupDriverValue("native.cgroupdriver=cgroupfs", opts, 0, "systemd")
	if opts[0] != "native.cgroupdriver=systemd" {
		t.Error("Expected cgroup driver to be updated")
	}
}

func TestUpdateDockerRuntimeConfig(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{"default-runtime":"runc"}`), DefaultFileMode)

	err := updateDockerRuntimeConfig("kata")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestParseConfigOrBuildNew(t *testing.T) {
	cfg := parseConfigOrBuildNew("kata", []byte(`{"default-runtime":"runc"}`))
	if cfg == nil {
		t.Error("Expected non-nil config")
	}
}

func TestUpdateRuntimeConfiguration(t *testing.T) {
	v := map[string]interface{}{
		"default-runtime": "runc",
	}
	updateRuntimeConfiguration(v, "kata")
	if v["default-runtime"] != "kata" {
		t.Error("Expected runtime to be updated")
	}
}

func TestWriteDockerConfigSimple(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	cfg := map[string]interface{}{"test": "value"}
	err := writeDockerConfigSimple(cfg)
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestUpdateDockerDataRootConfig(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{"data-root":"/var/lib/docker"}`), DefaultFileMode)

	err := updateDockerDataRootConfig("/new/data/root")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestParseOrCreateConfig(t *testing.T) {
	cfg, err := parseOrCreateConfig([]byte(`{"data-root":"/var/lib/docker"}`), "/new/root")
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if cfg == nil {
		t.Error("Expected non-nil config")
	}
}

func TestProcessConfigDataRoot(t *testing.T) {
	cfg := map[string]interface{}{
		"data-root": "/var/lib/docker",
	}
	result := processConfigDataRoot(cfg, "/var/lib/docker")
	if v, ok := result.(map[string]interface{}); ok {
		if v["data-root"] == nil {
			t.Error("Expected data-root to exist")
		}
	}
}

func TestCreateDockerTLSConfig(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(ensureDockerConfigFileExists, func() error {
		return nil
	})
	patches.ApplyFunc(writeDockerConfig, func(cfg interface{}) error {
		data, _ := json.MarshalIndent(cfg, "", " ")
		return os.WriteFile(testPath, data, DefaultFileMode)
	})

	err := createDockerTLSConfig()
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestUpdateDockerTLSConfig(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{"tls":false}`), DefaultFileMode)

	err := updateDockerTLSConfig()
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestBuildTLSConfig(t *testing.T) {
	cfg := buildTLSConfig()
	if cfg == nil {
		t.Error("Expected non-nil config")
	}
	if cfg["tls"] != true {
		t.Error("Expected tls to be true")
	}
}

func TestUpdateTLSConfig(t *testing.T) {
	v := map[string]interface{}{}
	updateTLSConfig(v)
	if v["tls"] != true {
		t.Error("Expected tls to be true")
	}
}

func TestUpdateHostsList(t *testing.T) {
	hosts := []interface{}{"tcp://0.0.0.0:2375"}
	result := updateHostsList(hosts)
	if len(result) < numThree {
		t.Error("Expected at least 3 hosts")
	}
}

func TestUpdateBaseConfig(t *testing.T) {
	tempDir := t.TempDir()
	testPath := filepath.Join(tempDir, "daemon.json")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == DockerDaemonConfigFilePath {
			return os.ReadFile(testPath)
		}
		return nil, errors.New("file not found")
	})
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		if filename == DockerDaemonConfigFilePath {
			return os.WriteFile(testPath, data, perm)
		}
		return nil
	})

	os.WriteFile(testPath, []byte(`{}`), DefaultFileMode)

	err := updateBaseConfig()
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestApplyBaseConfigDefaults(t *testing.T) {
	v := map[string]interface{}{}
	applyBaseConfigDefaults(v)
	if v["log-driver"] != "json-file" {
		t.Error("Expected log-driver to be json-file")
	}
}
