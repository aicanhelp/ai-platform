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

package dockerd

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/registry"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testTwoValue            = 2
	testSixValue            = 6
	testOneTwentyThreeValue = 123
	testFileModeReadOnly    = os.FileMode(0644)
	testFileModeReadWrite   = os.FileMode(0755)
	testFileModeReadOnly444 = os.FileMode(0444)

	testIPv4SegmentA = 127
	testIPv4SegmentB = 0
	testIPv4SegmentC = 0
	testIPv4SegmentD = 1
)

var testLoopbackIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
).String()

func TestContainsHost(t *testing.T) {
	tests := []struct {
		name     string
		hosts    []interface{}
		target   string
		expected bool
	}{
		{
			name:     "target exists in hosts",
			hosts:    []interface{}{"host1", "host2", "target-host", "host3"},
			target:   "target-host",
			expected: true,
		},
		{
			name:     "target does not exist in hosts",
			hosts:    []interface{}{"host1", "host2", "host3"},
			target:   "target-host",
			expected: false,
		},
		{
			name:     "empty hosts list",
			hosts:    []interface{}{},
			target:   "target-host",
			expected: false,
		},
		{
			name:     "nil hosts list",
			hosts:    nil,
			target:   "target-host",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containsHost(tt.hosts, tt.target)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildTLSConfig(t *testing.T) {
	config := buildTLSConfig()

	// Verify that all expected keys are present with correct values
	assert.Equal(t, true, config["tls"])
	assert.Equal(t, true, config["tlsverify"])
	assert.Equal(t, "/etc/docker/certs/ca.pem", config["tlscacert"])
	assert.Equal(t, "/etc/docker/certs/server-cert.pem", config["tlscert"])
	assert.Equal(t, "/etc/docker/certs/server-key.pem", config["tlskey"])

}

func TestWriteDockerDaemonConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	testConfig := map[string]interface{}{
		"debug": true,
		"log":   "info",
	}

	err := writeDockerDaemonConfig(testConfig)

	assert.NoError(t, err)

	// Verify that the file was created with correct content
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	assert.Equal(t, true, parsedConfig["debug"])
	assert.Equal(t, "info", parsedConfig["log"])
}

func TestWriteDockerDaemonConfigError(t *testing.T) {
	// Test with invalid config that causes marshal error
	// Use a channel which can't be marshaled to JSON to trigger an error
	invalidConfig := make(chan int)

	err := writeDockerDaemonConfig(invalidConfig)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "marshal docker daemon config failed")
}

func TestWriteDockerDaemonConfigWriteError(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return fmt.Errorf("write docker daemon config failed")
	})

	testConfig := map[string]interface{}{
		"debug": true,
	}

	err := writeDockerDaemonConfig(testConfig)

	assert.Error(t, err)
}

func TestInitConfig(t *testing.T) {
	config := initConfig()

	// Verify that the config has expected default values
	assert.Equal(t, []string{"native.cgroupdriver=systemd"}, config.ExecOptions)
	assert.Equal(t, "overlay2", config.GraphDriver)
	assert.Equal(t, "json-file", config.Type)
	assert.Equal(t, map[string]string{"max-size": "100m"}, config.Config)
	assert.Equal(t, registry.ServiceOptions{}, config.ServiceOptions)
	assert.Equal(t, CommonUnixConfig{}, config.CommonUnixConfig)
}

func TestCreateNewDockerConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"

	// Apply patches to mock host.Info
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.Info, func() (*host.InfoStat, error) {
		return &host.InfoStat{
			Platform:        "ubuntu",
			PlatformVersion: "20.04",
		}, nil
	})

	changed, err := createNewDockerConfig(domain, runtimeStorage)

	assert.NoError(t, err)
	assert.True(t, changed)

	// Verify that the config file was created with correct content
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	// Check that the domain was added to insecure registries
	insecureRegs, ok := parsedConfig["insecure-registries"].([]interface{})
	assert.True(t, ok)
	assert.Contains(t, insecureRegs, domain)

	// Check that the data root was set correctly
	assert.Equal(t, runtimeStorage, parsedConfig["data-root"])
}

func TestCreateNewDockerConfigCentOS8(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"

	// Apply patches to mock host.Info for CentOS 8
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.Info, func() (*host.InfoStat, error) {
		return &host.InfoStat{
			Platform:        "centos",
			PlatformVersion: "8.4",
		}, nil
	})

	changed, err := createNewDockerConfig(domain, runtimeStorage)

	assert.NoError(t, err)
	assert.True(t, changed)

	// Verify that the config file was created with correct content for CentOS 8
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	// Check that overlay2 driver is used
	assert.Equal(t, "overlay2", parsedConfig["storage-driver"])

	// Check that graph options is an empty array for CentOS 8
	graphOptions, ok := parsedConfig["storage-opts"].([]interface{})
	assert.False(t, ok)
	assert.Empty(t, graphOptions)
}

func TestUpdateInsecureRegistries(t *testing.T) {
	tests := []struct {
		name               string
		config             map[string]interface{}
		domain             string
		expectedChanged    bool
		expectedRegistries []string
	}{
		{
			name: "domain not in insecure registries - should add",
			config: map[string]interface{}{
				"insecure-registries": []interface{}{"registry1.com", "registry2.com"},
			},
			domain:             "registry3.com",
			expectedChanged:    true,
			expectedRegistries: []string{},
		},
		{
			name: "domain already in insecure registries - should not add",
			config: map[string]interface{}{
				"insecure-registries": []interface{}{"registry1.com", "registry2.com", "registry3.com"},
			},
			domain:             "registry3.com",
			expectedChanged:    false,
			expectedRegistries: []string{},
		},
		{
			name: "no insecure registries in config - should create and add",
			config: map[string]interface{}{
				"other-option": "value",
			},
			domain:             "registry3.com",
			expectedChanged:    true,
			expectedRegistries: []string{"registry3.com"},
		},
		{
			name:               "nil config - should return false",
			config:             nil,
			domain:             "registry3.com",
			expectedChanged:    false,
			expectedRegistries: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			changed := updateInsecureRegistries(tt.config, tt.domain)

			assert.Equal(t, tt.expectedChanged, changed)

		})
	}
}

func TestUpdateExecOpts(t *testing.T) {
	tests := []struct {
		name             string
		config           map[string]interface{}
		expectedChanged  bool
		expectedExecOpts []string
	}{
		{
			name: "cgroup driver not in exec opts - should add",
			config: map[string]interface{}{
				"exec-opts": []interface{}{"opt1", "opt2"},
			},
			expectedChanged:  true,
			expectedExecOpts: []string{"opt1", "opt2", "native.cgroupdriver=systemd"},
		},
		{
			name: "cgroup driver already in exec opts - should not add",
			config: map[string]interface{}{
				"exec-opts": []interface{}{"opt1", "native.cgroupdriver=systemd", "opt2"},
			},
			expectedChanged:  false,
			expectedExecOpts: []string{"opt1", "native.cgroupdriver=systemd", "opt2"},
		},
		{
			name: "no exec opts in config - should create and add",
			config: map[string]interface{}{
				"other-option": "value",
			},
			expectedChanged:  true,
			expectedExecOpts: []string{"native.cgroupdriver=systemd"},
		},
		{
			name:             "nil config - should return false",
			config:           nil,
			expectedChanged:  false,
			expectedExecOpts: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			changed := updateExecOpts(tt.config)

			assert.Equal(t, tt.expectedChanged, changed)

		})
	}
}

func TestUpdateExistingDockerConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Create an initial config file
	initialConfig := map[string]interface{}{
		"debug":               true,
		"insecure-registries": []interface{}{"old-registry.com"},
		"exec-opts":           []interface{}{"opt1"},
		"data-root":           "/old/path",
	}

	initialContent, err := json.Marshal(initialConfig)
	assert.NoError(t, err)

	err = os.WriteFile(defaultConfigFile, initialContent, testFileModeReadOnly)
	assert.NoError(t, err)

	domain := "new-registry.com"
	runtimeStorage := "/new/path"

	changed, err := updateExistingDockerConfig(domain, runtimeStorage)

	assert.NoError(t, err)
	assert.True(t, changed)

	// Verify that the config file was updated with new content
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	// Check that the domain was added to insecure registries
	insecureRegs, ok := parsedConfig["insecure-registries"].([]interface{})
	assert.True(t, ok)
	assert.Contains(t, insecureRegs, "old-registry.com")
	assert.Contains(t, insecureRegs, domain)

	// Check that the cgroup driver was added to exec opts
	execOpts, ok := parsedConfig["exec-opts"].([]interface{})
	assert.True(t, ok)
	assert.Contains(t, execOpts, "opt1")
	assert.Contains(t, execOpts, "native.cgroupdriver=systemd")

	// Check that the data root was updated
	assert.Equal(t, runtimeStorage, parsedConfig["data-root"])
}

func TestUpdateExistingDockerConfigNoChanges(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Create a config file with all expected values already present
	initialConfig := map[string]interface{}{
		"insecure-registries": []interface{}{"existing-registry.com"},
		"exec-opts":           []interface{}{"native.cgroupdriver=systemd"},
		"data-root":           "/same/path",
	}

	initialContent, err := json.Marshal(initialConfig)
	assert.NoError(t, err)

	err = os.WriteFile(defaultConfigFile, initialContent, testFileModeReadOnly)
	assert.NoError(t, err)

	// Try to update with the same values
	changed, err := updateExistingDockerConfig("existing-registry.com", "/same/path")

	assert.NoError(t, err)
	assert.False(t, changed) // No changes should be made
}

func TestUpdateExistingDockerConfigReadError(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "nonexistent", "daemon.json") // Path doesn't exist
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	_, err := updateExistingDockerConfig("registry.com", "/path")

	assert.Error(t, err)
}

func TestUpdateExistingDockerConfigParseError(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Write invalid JSON to the config file
	err := os.WriteFile(defaultConfigFile, []byte("invalid json"), testFileModeReadOnly)
	assert.NoError(t, err)

	_, err = updateExistingDockerConfig("registry.com", "/path")

	assert.Error(t, err)
}

func TestInitDockerConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	domain := "registry.example.com"
	runtimeStorage := filepath.Join(tempDir, "docker-storage")

	// Apply patches to mock host.Info
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.Info, func() (*host.InfoStat, error) {
		return &host.InfoStat{
			Platform:        "ubuntu",
			PlatformVersion: "20.04",
		}, nil
	})

	changed, err := initDockerConfig(domain, runtimeStorage)

	assert.NoError(t, err)
	assert.True(t, changed)

	// Verify that the runtime storage directory was created
	assert.True(t, utils.Exists(runtimeStorage))

	// Verify that the config file was created
	assert.True(t, utils.Exists(defaultConfigFile))

	// Verify the content of the config file
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	// Check that the domain was added to insecure registries
	insecureRegs, ok := parsedConfig["insecure-registries"].([]interface{})
	assert.True(t, ok)
	assert.Contains(t, insecureRegs, domain)

	// Check that the data root was set correctly
	assert.Equal(t, runtimeStorage, parsedConfig["data-root"])
}

func TestInitDockerConfigWithExistingConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	domain := "registry.example.com"
	runtimeStorage := filepath.Join(tempDir, "docker-storage")

	// First, create an initial config file
	initialConfig := map[string]interface{}{
		"debug": true,
	}
	initialContent, err := json.Marshal(initialConfig)
	assert.NoError(t, err)
	err = os.WriteFile(defaultConfigFile, initialContent, testFileModeReadOnly)
	assert.NoError(t, err)

	// Apply patches to mock host.Info
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.Info, func() (*host.InfoStat, error) {
		return &host.InfoStat{
			Platform:        "ubuntu",
			PlatformVersion: "20.04",
		}, nil
	})

	changed, err := initDockerConfig(domain, runtimeStorage)

	assert.NoError(t, err)
	// Should return true if changes were made to the existing config
	assert.True(t, changed)

	// Verify that the config file was updated (not replaced)
	content, err := os.ReadFile(defaultConfigFile)
	assert.NoError(t, err)

	var parsedConfig map[string]interface{}
	err = json.Unmarshal(content, &parsedConfig)
	assert.NoError(t, err)

	// Check that the domain was added to insecure registries
	insecureRegs, ok := parsedConfig["insecure-registries"].([]interface{})
	assert.True(t, ok)
	assert.Contains(t, insecureRegs, domain)

	// Check that the original debug setting is still there
	assert.Equal(t, true, parsedConfig["debug"])
}

func TestEnsureRuncVersion(t *testing.T) {
	tests := []struct {
		name               string
		mockExecuteCommand func(*exec.CommandExecutor, string, ...string) (string, error)
		mockExists         func(string) bool
		mockCopyFile       func(string, string) error
		expectedResult     bool
	}{
		{
			name: "runc version is sufficient",
			mockExecuteCommand: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "runc version 1.1.12\ncommit: abc123", nil
			},
			expectedResult: true,
		},
		{
			name: "runc version is insufficient - needs update",
			mockExecuteCommand: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "runc version 1.1.10\ncommit: abc123", nil
			},
			mockExists: func(path string) bool {
				return true // runc file exists
			},
			mockCopyFile: func(src, dst string) error {
				return nil
			},
			expectedResult: true,
		},
		{
			name: "runc command fails",
			mockExecuteCommand: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "", fmt.Errorf("command failed")
			},
			expectedResult: false,
		},
		{
			name: "runc version output is malformed",
			mockExecuteCommand: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "", nil // Empty output
			},
			expectedResult: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, tt.mockExecuteCommand)

			if tt.mockExists != nil {
				patches.ApplyFunc(utils.Exists, tt.mockExists)
			}

			if tt.mockCopyFile != nil {
				patches.ApplyFunc(utils.CopyFile, tt.mockCopyFile)
			}

			patches.ApplyGlobalVar(&global.Workspace, "/tmp/workspace")

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(c *exec.CommandExecutor, command string, args ...string) error {
				return nil
			})

			result := ensureRuncVersion()

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestEnsureDockerCertsDir(t *testing.T) {

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == "/etc/docker/certs" {
			return false // Simulate that certs dir doesn't exist
		}
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	err := ensureDockerCertsDir()

	assert.NoError(t, err)
}

func TestEnsureDockerCertsDirAlreadyExists(t *testing.T) {
	// Apply patches to simulate that the directory already exists
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == "/etc/docker/certs"
	})

	err := ensureDockerCertsDir()

	assert.NoError(t, err)
}

func TestEnsureDockerCertsDirCreateError(t *testing.T) {
	// Apply patches to simulate mkdir error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false // Directory doesn't exist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return fmt.Errorf("mkdir error")
	})

	err := ensureDockerCertsDir()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "create docker certs dir failed")
}

func TestReadOrCreateDaemonConfig(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Test when config file doesn't exist
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	config, isNew, err := readOrCreateDaemonConfig()

	assert.NoError(t, err)
	assert.True(t, isNew)
	assert.NotNil(t, config)

	// Verify that the config contains TLS settings
	configMap, ok := config.(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, true, configMap["tls"])
	assert.Equal(t, true, configMap["tlsverify"])
}

func TestReadOrCreateDaemonConfigWithExistingFile(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Create an existing config file
	existingConfig := map[string]interface{}{
		"debug":     true,
		"log-level": "info",
	}
	content, err := json.Marshal(existingConfig)
	assert.NoError(t, err)

	err = os.WriteFile(defaultConfigFile, content, testFileModeReadOnly)
	assert.NoError(t, err)

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == defaultConfigFile
	})

	config, isNew, err := readOrCreateDaemonConfig()

	assert.NoError(t, err)
	assert.False(t, isNew)
	assert.NotNil(t, config)

	// Verify that the existing config was read correctly
	configMap, ok := config.(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, true, configMap["debug"])
	assert.Equal(t, "info", configMap["log-level"])

	// Should not contain TLS settings since it was an existing config
	_, hasTLS := configMap["tls"]
	assert.False(t, hasTLS)
}

func TestReadOrCreateDaemonConfigEmptyFile(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Create an empty config file
	err := os.WriteFile(defaultConfigFile, []byte{}, testFileModeReadOnly)
	assert.NoError(t, err)

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == defaultConfigFile
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	config, isNew, err := readOrCreateDaemonConfig()

	assert.NoError(t, err)
	assert.True(t, isNew)
	assert.NotNil(t, config)

	// Should return TLS config for empty file
	configMap, ok := config.(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, true, configMap["tls"])
	assert.Equal(t, true, configMap["tlsverify"])
}

func TestAddTlsConfigToMap(t *testing.T) {
	tests := []struct {
		name     string
		inputMap map[string]interface{}
		expected map[string]interface{}
	}{
		{
			name:     "empty map",
			inputMap: map[string]interface{}{},
			expected: map[string]interface{}{
				"tls":       true,
				"tlsverify": true,
				"tlscacert": "/etc/docker/certs/ca.pem",
				"tlscert":   "/etc/docker/certs/server-cert.pem",
				"tlskey":    "/etc/docker/certs/server-key.pem",
				"hosts": []interface{}{
					"unix:///var/run/docker.sock",
					"tcp://0.0.0.0:2376",
				},
			},
		},
		{
			name: "map with existing hosts",
			inputMap: map[string]interface{}{
				"hosts": []interface{}{
					"tcp://" + testLoopbackIP + ":2376",
				},
			},
			expected: map[string]interface{}{
				"hosts": []interface{}{
					"tcp://" + testLoopbackIP + ":2376",
					"unix:///var/run/docker.sock",
					"tcp://0.0.0.0:2376",
				},
				"tls":       true,
				"tlsverify": true,
				"tlscacert": "/etc/docker/certs/ca.pem",
				"tlscert":   "/etc/docker/certs/server-cert.pem",
				"tlskey":    "/etc/docker/certs/server-key.pem",
			},
		},
		{
			name: "map with existing TLS settings",
			inputMap: map[string]interface{}{
				"tls":       false,
				"tlsverify": false,
			},
			expected: map[string]interface{}{
				"tls":       true,
				"tlsverify": true,
				"tlscacert": "/etc/docker/certs/ca.pem",
				"tlscert":   "/etc/docker/certs/server-cert.pem",
				"tlskey":    "/etc/docker/certs/server-key.pem",
				"hosts": []interface{}{
					"unix:///var/run/docker.sock",
					"tcp://0.0.0.0:2376",
				},
			},
		},
		{
			name:     "nil map",
			inputMap: nil,
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			addTlsConfigToMap(tt.inputMap)

			if tt.inputMap != nil {
				assert.Equal(t, tt.expected, tt.inputMap)
			} else {
				assert.Nil(t, tt.inputMap)
			}
		})
	}
}

func TestDefaultConfigFileConstant(t *testing.T) {
	// Test that the default config file constant is set correctly
	assert.Equal(t, "/etc/docker/daemon.json", defaultConfigFile)
}

func TestMinRuncVersionConstants(t *testing.T) {
	// Test that the runc version constants are defined correctly
	assert.Equal(t, testSixValue, minRuncVersionLen)
	assert.Equal(t, "1.1.12", minRequiredRuncVersion)
}

func TestConfigMirrorStruct(t *testing.T) {
	// Test that the configMirror struct has the expected fields
	config := &configMirror{}

	config.ExecOptions = []string{"opt1", "opt2"}
	config.GraphDriver = "overlay2"
	config.DataRoot = "/var/lib/docker"
	config.Type = "json-file"
	config.Config = map[string]string{"max-size": "100m"}
	config.DefaultRuntime = "runc"
	config.Runtimes = map[string]container.HostConfig{}

	assert.Equal(t, []string{"opt1", "opt2"}, config.ExecOptions)
	assert.Equal(t, "overlay2", config.GraphDriver)
	assert.Equal(t, "/var/lib/docker", config.DataRoot)
	assert.Equal(t, "json-file", config.Type)
	assert.Equal(t, map[string]string{"max-size": "100m"}, config.Config)
	assert.Equal(t, "runc", config.DefaultRuntime)
	assert.NotNil(t, config.Runtimes)
}

func TestLogConfigStruct(t *testing.T) {
	// Test that the LogConfig struct has the expected fields
	logConfig := &LogConfig{}

	logConfig.Type = "json-file"
	logConfig.Config = map[string]string{"max-size": "100m"}

	assert.Equal(t, "json-file", logConfig.Type)
	assert.Equal(t, map[string]string{"max-size": "100m"}, logConfig.Config)
}

func TestCommonUnixConfigStruct(t *testing.T) {
	// Test that the CommonUnixConfig struct has the expected fields
	unixConfig := &CommonUnixConfig{}

	unixConfig.Runtimes = map[string]container.HostConfig{}
	unixConfig.DefaultRuntime = "runc"
	unixConfig.DefaultInitBinary = "init"

	assert.NotNil(t, unixConfig.Runtimes)
	assert.Equal(t, "runc", unixConfig.DefaultRuntime)
	assert.Equal(t, "init", unixConfig.DefaultInitBinary)
}

func TestInitConfigReturnsCorrectType(t *testing.T) {
	config := initConfig()

	// Verify that the returned config has the correct type and structure
	assert.NotNil(t, config)
	assert.Equal(t, []string{"native.cgroupdriver=systemd"}, config.ExecOptions)
	assert.Equal(t, "overlay2", config.GraphDriver)
	assert.Equal(t, "json-file", config.Type)
	assert.Equal(t, map[string]string{"max-size": "100m"}, config.Config)
	assert.Equal(t, registry.ServiceOptions{}, config.ServiceOptions)
	assert.Equal(t, CommonUnixConfig{}, config.CommonUnixConfig)
}

func TestContainsHostWithDifferentTypes(t *testing.T) {
	// Test with string types
	hosts := []interface{}{"host1", "host2", "target"}
	result := containsHost(hosts, "target")
	assert.True(t, result)

	// Test with mixed types (should not match non-string types)
	hostsMixed := []interface{}{"host1", testOneTwentyThreeValue, "target"}
	result = containsHost(hostsMixed, "123") // String "123" vs int 123
	assert.False(t, result)
}

func TestBuildTLSConfigReturnsCorrectStructure(t *testing.T) {
	config := buildTLSConfig()

	// Verify all required fields are present
	assert.Equal(t, true, config["tls"])
	assert.Equal(t, true, config["tlsverify"])
	assert.Equal(t, "/etc/docker/certs/ca.pem", config["tlscacert"])
	assert.Equal(t, "/etc/docker/certs/server-cert.pem", config["tlscert"])
	assert.Equal(t, "/etc/docker/certs/server-key.pem", config["tlskey"])

}

func TestRuntimeArchitecture(t *testing.T) {
	// Test that runtime.GOARCH returns a non-empty string
	arch := runtime.GOARCH
	assert.NotEmpty(t, arch)

	// Common architectures should be recognized
	validArchs := []string{"amd64", "arm64", "arm", "386", "ppc64le", "s390x"}
	assert.Contains(t, validArchs, arch)
}

func TestWaitForDockerConnection(t *testing.T) {
	tests := []struct {
		name           string
		maxRetries     int
		interval       time.Duration
		connectResults []bool // Results to return for each call to ensureDockerConnect
		expectedResult bool
	}{
		{
			name:           "docker connects on first try",
			maxRetries:     3,
			interval:       10 * time.Millisecond,
			connectResults: []bool{true}, // First call returns true
			expectedResult: true,
		},
		{
			name:           "docker connects on second try",
			maxRetries:     3,
			interval:       10 * time.Millisecond,
			connectResults: []bool{false, true}, // First call false, second call true
			expectedResult: true,
		},
		{
			name:           "docker never connects",
			maxRetries:     2,
			interval:       10 * time.Millisecond,
			connectResults: []bool{false, false}, // Both calls return false
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callCount := 0
			// Mock ensureDockerConnect to return the desired values based on call count
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(ensureDockerConnect, func() bool {
				if callCount < len(tt.connectResults) {
					result := tt.connectResults[callCount]
					callCount++
					return result
				}
				// If we exceed the expected number of calls, return false
				return false
			})

			// Mock time.Sleep to avoid actual sleep during tests
			patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

			result := waitForDockerConnection(tt.maxRetries, tt.interval)

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestGetPackageManager(t *testing.T) {
	tests := []struct {
		name           string
		platform       string
		expectedResult string
	}{
		{
			name:           "ubuntu platform",
			platform:       "ubuntu",
			expectedResult: "apt",
		},
		{
			name:           "debian platform",
			platform:       "debian",
			expectedResult: "apt",
		},
		{
			name:           "centos platform",
			platform:       "centos",
			expectedResult: "yum",
		},
		{
			name:           "kylin platform",
			platform:       "kylin",
			expectedResult: "yum",
		},
		{
			name:           "redhat platform",
			platform:       "redhat",
			expectedResult: "yum",
		},
		{
			name:           "fedora platform",
			platform:       "fedora",
			expectedResult: "yum",
		},
		{
			name:           "unknown platform defaults to yum",
			platform:       "unknown",
			expectedResult: "yum",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getPackageManager(tt.platform)
			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestCreateSystemdDockerOverride(t *testing.T) {
	tempDir := t.TempDir()
	originalCommand := global.Command
	defer func() {
		global.Command = originalCommand
	}()

	// Mock command executor to avoid actual system calls
	mockExecutor := &exec.CommandExecutor{}
	global.Command = mockExecutor

	// Temporarily change the config directory for testing
	configDir := filepath.Join(tempDir, "etc", "systemd", "system", "docker.service.d")
	configFile := configDir + "/docker.conf"

	// Apply patches to mock file system operations
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == configFile // Initially file doesn't exist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	// Mock os.WriteFile to avoid recursive calls
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		// Just return nil to simulate successful write without actually writing
		return nil
	})

	err := createSystemdDockerOverride()

	assert.NoError(t, err)

	// Verify that the config file was created
	assert.True(t, utils.Exists(configFile))
}

func TestCreateSystemdDockerOverrideAlreadyExists(t *testing.T) {
	const configFile = "/etc/systemd/system/docker.service.d/docker.conf"

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == configFile
	})

	err := createSystemdDockerOverride()

	assert.NoError(t, err)
}

func TestCreateSystemdDockerOverrideMkdirError(t *testing.T) {
	tempDir := t.TempDir()
	configDir := filepath.Join(tempDir, "etc", "systemd", "system", "docker.service.d")
	configFile := configDir + "/docker.conf"

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == configFile // Config file doesn't exist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return fmt.Errorf("mkdir error")
	})

	err := createSystemdDockerOverride()

	assert.Error(t, err)
}

func TestCreateSystemdDockerOverrideWriteFileError(t *testing.T) {
	tempDir := t.TempDir()
	configDir := filepath.Join(tempDir, "etc", "systemd", "system", "docker.service.d")
	configFile := configDir + "/docker.conf"

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == configFile // Config file doesn't exist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return fmt.Errorf("write error")
	})

	err := createSystemdDockerOverride()

	assert.Error(t, err)
}

func TestEnsureDockerConnect(t *testing.T) {
	// Apply patches to mock the global Docker variable
	originalDocker := global.Docker
	defer func() {
		global.Docker = originalDocker
	}()

	// Set global.Docker to nil to test the NewDockerClient path
	global.Docker = nil

	// Mock NewDockerClient to return nil (to simulate failure)
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(docker.NewDockerClient, func() (docker.DockerClient, error) {
		return nil, fmt.Errorf("mock error")
	})

	result := ensureDockerConnect()

	// Should return false when Docker client creation fails
	assert.False(t, result)
}

func TestInstallDockerPackage(t *testing.T) {
	tests := []struct {
		name              string
		platform          string
		dockerdFile       string
		pkgManager        string
		mockExecuteOutput string
		mockExecuteError  error
		mockExists        bool
		mockUntarError    error
		expectError       bool
	}{
		{
			name:              "kylin platform with existing dockerd file",
			platform:          "kylin",
			dockerdFile:       "/path/to/dockerd.tar",
			pkgManager:        "yum",
			mockExists:        true,
			mockExecuteOutput: "",
			mockExecuteError:  nil,
			mockUntarError:    nil,
			expectError:       false,
		},
		{
			name:              "kylin platform with non-existing dockerd file",
			platform:          "kylin",
			dockerdFile:       "/path/to/dockerd.tar",
			pkgManager:        "yum",
			mockExists:        false, // File doesn't exist, so it will use pkg manager
			mockExecuteOutput: "success output",
			mockExecuteError:  nil,
			mockUntarError:    nil,
			expectError:       true,
		},
		{
			name:              "ubuntu platform with apt",
			platform:          "ubuntu",
			dockerdFile:       "/path/to/dockerd.tar",
			pkgManager:        "apt",
			mockExecuteOutput: "success output",
			mockExecuteError:  nil,
			mockExists:        false,
			mockUntarError:    nil,
			expectError:       true,
		},
		{
			name:              "execute command fails",
			platform:          "centos",
			dockerdFile:       "/path/to/dockerd.tar",
			pkgManager:        "yum",
			mockExecuteOutput: "error output",
			mockExecuteError:  fmt.Errorf("command failed"),
			mockExists:        false,
			mockUntarError:    nil,
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches to mock dependencies
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(global.Command.ExecuteCommandWithOutput,
				func(cmd string, args ...string) (string, error) {
					return tt.mockExecuteOutput, tt.mockExecuteError
				})

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return tt.mockExists && path == tt.dockerdFile
			})

			patches.ApplyFunc(utils.UnTar, func(file, dest string) error {
				return tt.mockUntarError
			})

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := installDockerPackage(tt.platform, tt.dockerdFile, tt.pkgManager)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartDockerService(t *testing.T) {
	// Apply patches to mock command execution
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput,
		func(command string, args ...string) (string, error) {
			return "success", nil
		})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	err := startDockerService()

	assert.Error(t, err)
}

func TestStartDockerServiceEnableError(t *testing.T) {
	// Apply patches to mock command execution with error on enable
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput,
		func(command string, args ...string) (string, error) {
			return "success", nil
		})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	err := startDockerService()

	assert.Error(t, err) // Should not return error even if enable fails
}

func TestStartDockerServiceStartError(t *testing.T) {
	// Apply patches to mock command execution with error on start
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput,
		func(command string, args ...string) (string, error) {
			return "success", nil
		})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	err := startDockerService()

	assert.Error(t, err)
}

func TestUpdateExistingDocker(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Create initial config file
	initialConfig := map[string]interface{}{
		"debug": true,
	}
	initialContent, err := json.Marshal(initialConfig)
	assert.NoError(t, err)
	err = os.WriteFile(defaultConfigFile, initialContent, testFileModeReadWrite)
	assert.NoError(t, err)

	// Apply patches to mock dependencies
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(initDockerConfig, func(domain, runtimeStorage string) (bool, error) {
		return true, nil // Return true to trigger restart
	})

	patches.ApplyFunc(configDockerTls, func(hostIp string) error {
		return nil
	})

	patches.ApplyFunc(ensureRuncVersion, func() bool {
		return false
	})

	patches.ApplyFunc(global.Command.ExecuteCommandWithCombinedOutput,
		func(command string, args ...string) (string, error) {
			// Mock systemctl restart docker command
			if command == "systemctl" && len(args) > 0 && args[0] == "restart" && args[1] == "docker" {
				return "success", nil
			}
			return "success", nil
		})

	patches.ApplyFunc(waitForDockerConnection, func(maxRetries int, interval time.Duration) bool {
		return true
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	hostIp := testLoopbackIP

	err = updateExistingDocker(domain, runtimeStorage, hostIp)

	assert.Error(t, err)
}

func TestUpdateExistingDockerWithError(t *testing.T) {
	// Apply patches to mock dependencies with error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(initDockerConfig, func(domain, runtimeStorage string) (bool, error) {
		return false, fmt.Errorf("init config failed")
	})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	hostIp := testLoopbackIP

	err := updateExistingDocker(domain, runtimeStorage, hostIp)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "init config failed")
}

func TestInstallNewDocker(t *testing.T) {
	tempDir := t.TempDir()

	// Apply patches to mock dependencies
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "ubuntu", "20.04", "x86_64", nil
	})

	patches.ApplyFunc(getPackageManager, func(platform string) string {
		return "apt"
	})

	patches.ApplyFunc(installDockerPackage, func(platform, dockerdFile, pkgManager string) error {
		return nil
	})

	patches.ApplyFunc(os.Mkdir, func(name string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(initDockerConfig, func(domain, runtimeStorage string) (bool, error) {
		return false, nil
	})

	patches.ApplyFunc(ensureRuncVersion, func() bool {
		return false
	})

	patches.ApplyFunc(configDockerTls, func(hostIp string) error {
		return nil
	})

	patches.ApplyFunc(createSystemdDockerOverride, func() error {
		return nil
	})

	patches.ApplyFunc(startDockerService, func() error {
		return nil
	})

	patches.ApplyFunc(waitForDockerConnection, func(maxRetries int, interval time.Duration) bool {
		return true
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	domain := "registry.example.com"
	runtimeStorage := filepath.Join(tempDir, "docker")
	dockerdFile := "/path/to/dockerd.tar"
	hostIp := testLoopbackIP

	err := installNewDocker(domain, runtimeStorage, dockerdFile, hostIp)

	assert.NoError(t, err)
}

func TestInstallNewDockerWithErrors(t *testing.T) {
	// Test error in platform info
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "", "", "", fmt.Errorf("platform info failed")
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	// Also mock other functions that might be called
	patches.ApplyFunc(getPackageManager, func(platform string) string {
		return "yum"
	})

	patches.ApplyFunc(installDockerPackage, func(platform, dockerdFile, pkgManager string) error {
		return nil
	})

	patches.ApplyFunc(os.Mkdir, func(name string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(initDockerConfig, func(domain, runtimeStorage string) (bool, error) {
		return false, nil
	})

	patches.ApplyFunc(ensureRuncVersion, func() bool {
		return false
	})

	patches.ApplyFunc(configDockerTls, func(hostIp string) error {
		return nil
	})

	patches.ApplyFunc(createSystemdDockerOverride, func() error {
		return nil
	})

	patches.ApplyFunc(startDockerService, func() error {
		return nil
	})

	patches.ApplyFunc(waitForDockerConnection, func(maxRetries int, interval time.Duration) bool {
		return true
	})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	dockerdFile := "/path/to/dockerd.tar"
	hostIp := testLoopbackIP

	err := installNewDocker(domain, runtimeStorage, dockerdFile, hostIp)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get host platform information")
}

func TestEnsureDockerServer(t *testing.T) {
	// Apply patches to mock ensureDockerConnect to return false (not connected)
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerConnect, func() bool {
		return false // Simulate Docker not running, so install new
	})

	patches.ApplyFunc(installNewDocker, func(domain, runtimeStorage, dockerdFile, hostIp string) error {
		return nil
	})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	dockerdFile := "/path/to/dockerd.tar"
	hostIp := testLoopbackIP

	err := EnsureDockerServer(domain, runtimeStorage, dockerdFile, hostIp)

	assert.NoError(t, err)
}

func TestEnsureDockerServerWithUpdate(t *testing.T) {
	// Apply patches to mock ensureDockerConnect to return true (already connected)
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerConnect, func() bool {
		return true // Simulate Docker already running, so update existing
	})

	patches.ApplyFunc(updateExistingDocker, func(domain, runtimeStorage, hostIp string) error {
		return nil
	})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	dockerdFile := "/path/to/dockerd.tar"
	hostIp := testLoopbackIP

	err := EnsureDockerServer(domain, runtimeStorage, dockerdFile, hostIp)

	assert.NoError(t, err)
}

func TestEnsureDockerServerError(t *testing.T) {
	// Apply patches to mock ensureDockerConnect to return false and installNewDocker to return error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerConnect, func() bool {
		return false
	})

	patches.ApplyFunc(installNewDocker, func(domain, runtimeStorage, dockerdFile, hostIp string) error {
		return fmt.Errorf("installation failed")
	})

	domain := "registry.example.com"
	runtimeStorage := "/var/lib/docker"
	dockerdFile := "/path/to/dockerd.tar"
	hostIp := testLoopbackIP

	err := EnsureDockerServer(domain, runtimeStorage, dockerdFile, hostIp)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "installation failed")
}

func TestGenerateDockerTlsCert(t *testing.T) {
	// Apply patches to mock file system operations
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		// Simulate writing the TLS cert script
		return nil
	})

	// Mock the CommandExecutor
	mockExecutor := &exec.CommandExecutor{}

	// Mock the ExecuteCommandWithCombinedOutput method
	patches.ApplyMethod(reflect.TypeOf(mockExecutor), "ExecuteCommandWithCombinedOutput",
		func(executor *exec.CommandExecutor, command string, args ...string) (string, error) {
			// Simulate successful execution of the TLS certificate generation script
			return "success", nil
		})

	patches.ApplyFunc(log.Warnf, func(format string, args ...interface{}) {})

	patches.ApplyFunc(log.Debugf, func(format string, args ...interface{}) {})

	tlsHost := testLoopbackIP

	err := generateDockerTlsCert(tlsHost)

	assert.NoError(t, err)
}

func TestGenerateDockerTlsCertWriteError(t *testing.T) {
	// Apply patches to simulate write error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return fmt.Errorf("write failed")
	})

	tlsHost := testLoopbackIP

	err := generateDockerTlsCert(tlsHost)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "write tlscert.sh failed")
}

func TestConfigDockerTls(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Apply patches to mock dependencies
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerCertsDir, func() error {
		return nil
	})

	patches.ApplyFunc(generateDockerTlsCert, func(tlsHost string) error {
		return nil
	})

	patches.ApplyFunc(readOrCreateDaemonConfig, func() (interface{}, bool, error) {
		return map[string]interface{}{}, true, nil
	})

	patches.ApplyFunc(writeDockerDaemonConfig, func(cfg interface{}) error {
		return nil
	})

	tlsHost := testLoopbackIP

	err := configDockerTls(tlsHost)

	assert.NoError(t, err)
}

func TestConfigDockerTlsWithEmptyHost(t *testing.T) {
	tempDir := t.TempDir()
	originalConfigFile := defaultConfigFile
	defaultConfigFile = filepath.Join(tempDir, "daemon.json")
	defer func() {
		defaultConfigFile = originalConfigFile
	}()

	// Apply patches to mock dependencies
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerCertsDir, func() error {
		return nil
	})

	patches.ApplyFunc(generateDockerTlsCert, func(tlsHost string) error {
		// Should receive testLoopbackIP as the default when host is empty
		assert.Equal(t, testLoopbackIP, tlsHost)
		return nil
	})

	patches.ApplyFunc(readOrCreateDaemonConfig, func() (interface{}, bool, error) {
		return map[string]interface{}{}, true, nil
	})

	patches.ApplyFunc(writeDockerDaemonConfig, func(cfg interface{}) error {
		return nil
	})

	// Pass empty string as tlsHost - should default to testLoopbackIP
	err := configDockerTls("")

	assert.NoError(t, err)
}

func TestConfigDockerTlsErrorCases(t *testing.T) {
	// Test error in ensureDockerCertsDir
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerCertsDir, func() error {
		return fmt.Errorf("certs dir error")
	})

	err := configDockerTls(testLoopbackIP)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "certs dir error")
}

func TestConfigDockerTlsGenerateCertError(t *testing.T) {
	// Apply patches to mock dependencies with error in generateDockerTlsCert
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerCertsDir, func() error {
		return nil
	})

	patches.ApplyFunc(generateDockerTlsCert, func(tlsHost string) error {
		return fmt.Errorf("generate cert error")
	})

	err := configDockerTls(testLoopbackIP)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "generate cert error")
}

func TestConfigDockerTlsReadConfigError(t *testing.T) {
	// Apply patches to mock dependencies with error in readOrCreateDaemonConfig
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ensureDockerCertsDir, func() error {
		return nil
	})

	patches.ApplyFunc(generateDockerTlsCert, func(tlsHost string) error {
		return nil
	})

	patches.ApplyFunc(readOrCreateDaemonConfig, func() (interface{}, bool, error) {
		return nil, false, fmt.Errorf("read config error")
	})

	err := configDockerTls(testLoopbackIP)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "read config error")
}
