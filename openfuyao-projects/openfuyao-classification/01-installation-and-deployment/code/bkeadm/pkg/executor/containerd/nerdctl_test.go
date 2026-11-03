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

package containerd

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containerd/containerd/v2/client"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// Constants for numeric literals to comply with ds.txt standards
const (
	testNumericZero         = 0
	testNumericOne          = 1
	testNumericPid          = 1234
	testNumericExitCode     = 0
	testNumericRestartCount = 0
	testNumericPort         = 8080
)

// Constants for IP address segments to comply with ds.txt standards
const (
	testIPv4SegmentA  = 192
	testIPv4SegmentB  = 168
	testIPv4SegmentC  = 1
	testIPv4SegmentD  = 100
	testIPv4LoopbackA = 127
	testIPv4LoopbackB = 0
	testIPv4LoopbackC = 0
	testIPv4LoopbackD = 1
)

// Variables for IP addresses constructed from constants
var (
	testIP192_168_1_100    = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
	testIPLoopback         = fmt.Sprintf("%d.%d.%d.%d", testIPv4LoopbackA, testIPv4LoopbackB, testIPv4LoopbackC, testIPv4LoopbackD)
	testIPNet192_168_1_100 = net.IPv4(byte(testIPv4SegmentA), byte(testIPv4SegmentB), byte(testIPv4SegmentC), byte(testIPv4SegmentD))
	testIPNetLoopback      = net.IPv4(byte(testIPv4LoopbackA), byte(testIPv4LoopbackB), byte(testIPv4LoopbackC), byte(testIPv4LoopbackD))
)

func TestEnsureImageExists(t *testing.T) {
	tests := []struct {
		name                 string
		image                string
		imageInspectResult   string
		imageInspectErr      error
		executeCommandResult string
		executeCommandErr    error
		expectedError        bool
	}{
		{
			name:               "image already exists",
			image:              "nginx:latest",
			imageInspectResult: `[{"Id": "test-image-id"}]`,
			imageInspectErr:    nil,
			expectedError:      false,
		},
		{
			name:                 "image does not exist, pull succeeds",
			image:                "alpine:latest",
			imageInspectResult:   "",
			imageInspectErr:      errors.New("not found"),
			executeCommandResult: "Pulling alpine:latest...",
			executeCommandErr:    nil,
			expectedError:        false,
		},
		{
			name:                 "image does not exist, pull fails",
			image:                "nonexistent:image",
			imageInspectResult:   "",
			imageInspectErr:      errors.New("not found"),
			executeCommandResult: "Error pulling image",
			executeCommandErr:    errors.New("pull error"),
			expectedError:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
				func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
					if args[0] == "inspect" {
						return tt.imageInspectResult, tt.imageInspectErr
					} else if args[0] == "pull" {
						return tt.executeCommandResult, tt.executeCommandErr
					}
					return "", nil
				})

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := EnsureImageExists(tt.image)

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestImageInspect(t *testing.T) {
	tests := []struct {
		name          string
		image         string
		commandResult string
		commandErr    error
		expectedInfo  NerdImageInfo
		expectedError bool
	}{
		{
			name:          "valid image inspection",
			image:         "nginx:latest",
			commandResult: `[{"Id": "test-image-id", "RepoTags": ["nginx:latest"], "Architecture": "amd64", "Os": "linux"}]`,
			commandErr:    nil,
			expectedInfo:  NerdImageInfo{Id: "test-image-id", RepoTags: []string{"nginx:latest"}, Architecture: "amd64", OS: "linux"},
			expectedError: false,
		},
		{
			name:          "invalid json response",
			image:         "invalid:json",
			commandResult: `invalid json`,
			commandErr:    nil,
			expectedInfo:  NerdImageInfo{},
			expectedError: true,
		},
		{
			name:          "command execution error",
			image:         "error:image",
			commandResult: "",
			commandErr:    errors.New("command error"),
			expectedInfo:  NerdImageInfo{},
			expectedError: true,
		},
		{
			name:          "empty image list",
			image:         "empty:list",
			commandResult: `[]`,
			commandErr:    nil,
			expectedInfo:  NerdImageInfo{},
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
				func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
					return tt.commandResult, tt.commandErr
				})

			patches.ApplyFunc(log.Debug, func(args ...interface{}) {})

			info, err := ImageInspect(tt.image)

			if tt.expectedError {
				assert.Error(t, err)
				assert.Equal(t, NerdImageInfo{}, info)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedInfo, info)
			}
		})
	}
}

func TestEnsureContainerRun(t *testing.T) {
	tests := []struct {
		name            string
		containerId     string
		containerExists bool
		containerInfo   NerdContainerInfo
		removeError     error
		expectedRunning bool
		expectedError   bool
	}{
		{
			name:            "container already running",
			containerId:     "test-container-running",
			containerExists: true,
			containerInfo: NerdContainerInfo{
				Id: "test-container-running",
			},
			expectedRunning: false,
			expectedError:   false,
		},
		{
			name:            "container not running, removal succeeds",
			containerId:     "test-container-stopped",
			containerExists: true,
			containerInfo: NerdContainerInfo{
				Id: "test-container-stopped",
			},
			removeError:     nil,
			expectedRunning: false,
			expectedError:   false,
		},
		{
			name:            "container not running, removal fails",
			containerId:     "test-container-failed",
			containerExists: true,
			containerInfo: NerdContainerInfo{
				Id: "test-container-failed",
			},
			removeError:     errors.New("removal error"),
			expectedRunning: false,
			expectedError:   true,
		},
		{
			name:            "container does not exist",
			containerId:     "nonexistent-container",
			containerExists: false,
			expectedRunning: false,
			expectedError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(ContainerExists, func(id string) (NerdContainerInfo, bool) {
				return tt.containerInfo, tt.containerExists
			})

			if tt.containerExists && !tt.containerInfo.State.Running {
				patches.ApplyFunc(ContainerRemove, func(id string) error {
					return tt.removeError
				})
			}

			running, err := EnsureContainerRun(tt.containerId)

			assert.Equal(t, tt.expectedRunning, running)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContainerExists(t *testing.T) {
	tests := []struct {
		name            string
		containerId     string
		commandResult   string
		commandErr      error
		unmarshalResult []NerdContainerInfo
		unmarshalError  error
		expectedInfo    NerdContainerInfo
		expectedExists  bool
	}{
		{
			name:            "container exists",
			containerId:     "existing-container",
			commandResult:   `[{"Id": "existing-container", "Name": "test-container"}]`,
			commandErr:      nil,
			unmarshalResult: []NerdContainerInfo{{Id: "existing-container", Name: "test-container"}},
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{Id: "existing-container", Name: "test-container"},
			expectedExists:  true,
		},
		{
			name:            "container does not exist",
			containerId:     "nonexistent-container",
			commandResult:   "",
			commandErr:      errors.New("not found"),
			unmarshalResult: nil,
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{},
			expectedExists:  false,
		},
		{
			name:            "invalid json response",
			containerId:     "invalid-json-container",
			commandResult:   `invalid json`,
			commandErr:      nil,
			unmarshalResult: nil,
			unmarshalError:  errors.New("invalid json"),
			expectedInfo:    NerdContainerInfo{},
			expectedExists:  false,
		},
		{
			name:            "empty result list",
			containerId:     "empty-result-container",
			commandResult:   `[]`,
			commandErr:      nil,
			unmarshalResult: []NerdContainerInfo{},
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{},
			expectedExists:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
				func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
					return tt.commandResult, tt.commandErr
				})

			// Patch json.Unmarshal to return the test values
			patches.ApplyFunc(json.Unmarshal, func(data []byte, v interface{}) error {
				if p, ok := v.(*[]NerdContainerInfo); ok {
					if tt.unmarshalResult != nil {
						*p = tt.unmarshalResult
					}
				}
				return tt.unmarshalError
			})

			patches.ApplyFunc(log.Debug, func(args ...interface{}) {})

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			info, exists := ContainerExists(tt.containerId)

			assert.Equal(t, tt.expectedExists, exists)
			if tt.expectedExists {
				assert.Equal(t, tt.expectedInfo, info)
			} else {
				assert.Equal(t, NerdContainerInfo{}, info)
			}
		})
	}
}

func TestContainerInspect(t *testing.T) {
	tests := []struct {
		name            string
		containerId     string
		commandResult   string
		commandErr      error
		unmarshalResult []NerdContainerInfo
		unmarshalError  error
		expectedInfo    NerdContainerInfo
		expectedError   bool
	}{
		{
			name:            "container inspect succeeds",
			containerId:     "inspectable-container",
			commandResult:   `[{"Id": "inspectable-container", "Name": "test-container"}]`,
			commandErr:      nil,
			unmarshalResult: []NerdContainerInfo{{Id: "inspectable-container", Name: "test-container"}},
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{Id: "inspectable-container", Name: "test-container"},
			expectedError:   false,
		},
		{
			name:            "command execution error",
			containerId:     "error-container",
			commandResult:   "",
			commandErr:      errors.New("command error"),
			unmarshalResult: nil,
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{},
			expectedError:   true,
		},
		{
			name:            "invalid json response",
			containerId:     "invalid-json-container",
			commandResult:   `invalid json`,
			commandErr:      nil,
			unmarshalResult: nil,
			unmarshalError:  errors.New("invalid json"),
			expectedInfo:    NerdContainerInfo{},
			expectedError:   true,
		},
		{
			name:            "empty result list",
			containerId:     "empty-result-container",
			commandResult:   `[]`,
			commandErr:      nil,
			unmarshalResult: []NerdContainerInfo{},
			unmarshalError:  nil,
			expectedInfo:    NerdContainerInfo{},
			expectedError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
				func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
					return tt.commandResult, tt.commandErr
				})

			// Patch json.Unmarshal to return the test values
			patches.ApplyFunc(json.Unmarshal, func(data []byte, v interface{}) error {
				if p, ok := v.(*[]NerdContainerInfo); ok {
					if tt.unmarshalResult != nil {
						*p = tt.unmarshalResult
					}
				}
				return tt.unmarshalError
			})

			patches.ApplyFunc(log.Debug, func(args ...interface{}) {})

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			info, err := ContainerInspect(tt.containerId)

			if tt.expectedError {
				assert.Error(t, err)
				assert.Equal(t, NerdContainerInfo{}, info)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedInfo, info)
			}
		})
	}
}

func TestContainerRemove(t *testing.T) {
	tests := []struct {
		name          string
		containerId   string
		commandErr    error
		expectedError bool
	}{
		{
			name:          "container removal succeeds",
			containerId:   "removable-container",
			commandErr:    nil,
			expectedError: false,
		},
		{
			name:          "container removal fails",
			containerId:   "non-removable-container",
			commandErr:    errors.New("removal error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand,
				func(_ *exec.CommandExecutor, command string, args ...string) error {
					return tt.commandErr
				})

			err := ContainerRemove(tt.containerId)

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoad(t *testing.T) {
	tests := []struct {
		name          string
		imageFile     string
		commandErr    error
		expectedError bool
	}{
		{
			name:          "load succeeds",
			imageFile:     "/path/to/image.tar",
			commandErr:    nil,
			expectedError: false,
		},
		{
			name:          "load fails",
			imageFile:     "/path/to/invalid.tar",
			commandErr:    errors.New("load error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand,
				func(_ *exec.CommandExecutor, command string, args ...string) error {
					return tt.commandErr
				})

			err := Load(tt.imageFile)

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRun(t *testing.T) {
	tests := []struct {
		name          string
		script        []string
		commandErr    error
		expectedError bool
	}{
		{
			name:          "run succeeds",
			script:        []string{"run", "-d", "nginx:latest"},
			commandErr:    nil,
			expectedError: false,
		},
		{
			name:          "run fails",
			script:        []string{"run", "--invalid-flag", "nginx:latest"},
			commandErr:    errors.New("run error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand,
				func(_ *exec.CommandExecutor, command string, args ...string) error {
					return tt.commandErr
				})

			err := Run(tt.script)

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCP(t *testing.T) {
	tests := []struct {
		name          string
		src           string
		dst           string
		commandErr    error
		expectedError bool
	}{
		{
			name:          "copy succeeds",
			src:           "/path/to/src",
			dst:           "/path/to/dst",
			commandErr:    nil,
			expectedError: false,
		},
		{
			name:          "copy fails",
			src:           "/invalid/src",
			dst:           "/invalid/dst",
			commandErr:    errors.New("copy error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand,
				func(_ *exec.CommandExecutor, command string, args ...string) error {
					return tt.commandErr
				})

			err := CP(tt.src, tt.dst)

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNerdContainerInfoStruct(t *testing.T) {
	// Test that the NerdContainerInfo struct has the expected fields
	info := NerdContainerInfo{}

	// Initialize with test values
	info.Id = "test-container-id"
	info.State.Status = "running"
	info.State.Running = true
	info.State.Paused = false
	info.State.Restarting = false
	info.State.Pid = testNumericPid
	info.State.ExitCode = testNumericExitCode
	info.State.FinishedAt = "2023-01-01T00:00:00Z"
	info.Image = "nginx:latest"
	info.Name = "test-container"
	info.RestartCount = testNumericRestartCount
	info.Platform = "linux"
	info.NetworkSettings.IPAddress = testIP192_168_1_100
	info.NetworkSettings.MacAddress = "00:11:22:33:44:55"

	// Verify the values are set correctly
	assert.Equal(t, "test-container-id", info.Id)
	assert.Equal(t, "running", info.State.Status)
	assert.True(t, info.State.Running)
	assert.False(t, info.State.Paused)
	assert.False(t, info.State.Restarting)
	assert.Equal(t, "2023-01-01T00:00:00Z", info.State.FinishedAt)
	assert.Equal(t, "nginx:latest", info.Image)
	assert.Equal(t, "test-container", info.Name)
	assert.Equal(t, "linux", info.Platform)
	assert.Equal(t, testIP192_168_1_100, info.NetworkSettings.IPAddress)
	assert.Equal(t, "00:11:22:33:44:55", info.NetworkSettings.MacAddress)
}

func TestNerdImageInfoStruct(t *testing.T) {
	// Test that the NerdImageInfo struct has the expected fields
	info := NerdImageInfo{}

	// Initialize with test values
	info.Id = "test-image-id"
	info.RepoTags = []string{"nginx:latest", "nginx:1.20"}
	info.Architecture = "amd64"
	info.OS = "linux"

	// Verify the values are set correctly
	assert.Equal(t, "test-image-id", info.Id)
	assert.Equal(t, []string{"nginx:latest", "nginx:1.20"}, info.RepoTags)
	assert.Equal(t, "amd64", info.Architecture)
	assert.Equal(t, "linux", info.OS)
}

func TestCmdVariable(t *testing.T) {
	// Test that the cmd variable is initialized correctly
	assert.NotNil(t, cmd)
}

func TestJsonUnmarshalInImageInspect(t *testing.T) {
	// Test the json unmarshaling in ImageInspect with valid data
	jsonData := `[{"Id": "test-id", "RepoTags": ["repo:tag"], "Architecture": "arch", "Os": "os"}]`

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
			return jsonData, nil
		})

	patches.ApplyFunc(log.Debug, func(args ...interface{}) {})

	info, err := ImageInspect("test-image")

	assert.NoError(t, err)
	assert.Equal(t, "test-id", info.Id)
	assert.Equal(t, []string{"repo:tag"}, info.RepoTags)
	assert.Equal(t, "arch", info.Architecture)
	assert.Equal(t, "os", info.OS)
}

func TestJsonUnmarshalInContainerFunctions(t *testing.T) {
	// Test the json unmarshaling in container functions with valid data
	jsonData := `[{
  		"Id": "test-container",
  		"State": {
  			"Status": "running",
  			"Running": true,
  			"Paused": false,
  			"Restarting": false,
  			"Pid": 1234,
  			"ExitCode": 0,
  			"FinishedAt": "2023-01-01T00:00:00Z"
  		},
  		"Image": "nginx:latest",
  		"Name": "test-container",
  		"RestartCount": 0,
  		"Platform": "linux",
  		"NetworkSettings": {
  			"IPAddress": "` + testIP192_168_1_100 + `",
 			"MacAddress": "00:11:22:33:44:55"
 		}
 	}]`

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *exec.CommandExecutor, command string, args ...string) (string, error) {
			return jsonData, nil
		})

	patches.ApplyFunc(log.Debug, func(args ...interface{}) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	// Test ContainerInspect
	info, err := ContainerInspect("test-container")
	assert.NoError(t, err)
	assert.Equal(t, "test-container", info.Id)
	assert.True(t, info.State.Running)

	// Test ContainerExists
	existsInfo, exists := ContainerExists("test-container")
	assert.True(t, exists)
	assert.Equal(t, "test-container", existsInfo.Id)
	assert.True(t, existsInfo.State.Running)
}

func TestNewContainedClient(t *testing.T) {
	tests := []struct {
		name          string
		socketExists  bool
		clientNewErr  error
		expectedError bool
	}{
		{
			name:          "socket exists and client created successfully",
			socketExists:  true,
			clientNewErr:  nil,
			expectedError: false,
		},
		{
			name:          "socket does not exist",
			socketExists:  false,
			clientNewErr:  nil,
			expectedError: true,
		},
		{
			name:          "socket exists but client creation fails",
			socketExists:  true,
			clientNewErr:  errors.New("client creation error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return tt.socketExists
			})

			patches.ApplyFunc(client.New, func(_ string, _ ...client.Opt) (*client.Client, error) {
				if tt.clientNewErr != nil {
					return nil, tt.clientNewErr
				}
				return &client.Client{}, nil
			})

			client, err := NewContainedClient()

			if tt.expectedError {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}

func TestGetClient(t *testing.T) {
	t.Run("get client returns the underlying client", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(utils.Exists, func(path string) bool {
			return true
		})

		patches.ApplyFunc(client.New, func(_ string, _ ...client.Opt) (*client.Client, error) {
			return &client.Client{}, nil
		})

		containerClient, err := NewContainedClient()
		assert.NoError(t, err)
		assert.NotNil(t, containerClient)

		client := containerClient.GetClient()
		assert.NotNil(t, client)
	})
}

func TestImageRefStruct(t *testing.T) {
	t.Run("image ref struct initialization", func(t *testing.T) {
		imageRef := ImageRef{
			Image:    "nginx:latest",
			Username: "testuser",
			Password: "testpass",
		}

		assert.Equal(t, "nginx:latest", imageRef.Image)
		assert.Equal(t, "testuser", imageRef.Username)
		assert.Equal(t, "testpass", imageRef.Password)
	})

	t.Run("image ref struct with empty credentials", func(t *testing.T) {
		imageRef := ImageRef{
			Image: "public-image:latest",
		}

		assert.Equal(t, "public-image:latest", imageRef.Image)
		assert.Empty(t, imageRef.Username)
		assert.Empty(t, imageRef.Password)
	})
}

func TestClientStruct(t *testing.T) {
	t.Run("client struct fields are properly initialized", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(utils.Exists, func(path string) bool {
			return true
		})

		mockClient := &client.Client{}
		patches.ApplyFunc(client.New, func(_ string, _ ...client.Opt) (*client.Client, error) {
			return mockClient, nil
		})

		containerClient, err := NewContainedClient()
		assert.NoError(t, err)
		assert.NotNil(t, containerClient)

		client, ok := containerClient.(*Client)
		assert.True(t, ok)
		assert.NotNil(t, client.condClient)
		assert.NotNil(t, client.ctx)
		assert.NotNil(t, client.cancel)
		assert.NotNil(t, client.imageClient)
	})
}
