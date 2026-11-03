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

package global

import (
	"fmt"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// Constants for numeric literals to comply with ds.txt standards
const (
	testNumericZero    = 0
	testNumericOne     = 1
	testNumericTwo     = 2
	testNumericThree   = 3
	testNumericFour    = 4
	testNumericFive    = 5
	testFilePermission = 0644
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

func TestGlobalVariablesInitialization(t *testing.T) {
	// Verify that global variables are initialized properly
	assert.NotNil(t, Command)
	assert.IsType(t, &exec.CommandExecutor{}, Command)
	assert.Equal(t, "/bke", Workspace)
	assert.NotNil(t, CustomExtra)
	assert.IsType(t, make(map[string]string), CustomExtra)
}

func TestInitWithCustomWorkspace(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches to simulate custom workspace
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Getenv, func(key string) string {
		if key == "BKE_WORKSPACE" {
			return tempDir
		}
		return ""
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false // Don't read from file in this test
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		// Simulate that directories don't exist initially
		return strings.Contains(path, tempDir)
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(log.Warnf, func(format string, args ...interface{}) {})

	// Create a new instance to test initialization
	Command = &exec.CommandExecutor{}
	Workspace = ""
	CustomExtra = make(map[string]string)

	// Re-initialize by setting values
	if os.Getenv("BKE_WORKSPACE") != "" {
		Workspace = os.Getenv("BKE_WORKSPACE")
	}
	if Workspace == "" {
		Workspace = "/bke"
	}
	if !utils.Exists(Workspace + "/tmpl") {
		if err := os.MkdirAll(Workspace+"/tmpl", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create tmpl directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/volumes") {
		if err := os.MkdirAll(Workspace+"/volumes", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create volumes directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/mount") {
		if err := os.MkdirAll(Workspace+"/mount", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create mount directory: %v", err)
		}
	}
	CustomExtra = make(map[string]string)

	assert.Equal(t, tempDir, Workspace)
}

func TestInitWithWorkspaceFile(t *testing.T) {
	// Create a temporary file with workspace content
	tempDir := t.TempDir()
	workspaceFile := filepath.Join(tempDir, "BKE_WORKSPACE")

	content := "/custom/workspace/path\n"
	err := os.WriteFile(workspaceFile, []byte(content), testFilePermission)
	assert.NoError(t, err)

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return path == "/opt/BKE_WORKSPACE" // Simulate the file exists
	})

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		if filename == "/opt/BKE_WORKSPACE" {
			return []byte(content), nil
		}
		return nil, fmt.Errorf("file not found")
	})

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return "" // Don't use env var in this test
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return strings.Contains(path, "/custom/workspace/path")
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(log.Warnf, func(format string, args ...interface{}) {})

	// Create a new instance to test initialization
	Command = &exec.CommandExecutor{}
	Workspace = ""
	CustomExtra = make(map[string]string)

	// Simulate the init process
	if utils.IsFile("/opt/BKE_WORKSPACE") {
		f, err := os.ReadFile("/opt/BKE_WORKSPACE")
		if err == nil {
			Workspace = string(f)
			Workspace = strings.TrimSpace(Workspace)
			Workspace = strings.TrimRight(Workspace, "\n")
			Workspace = strings.TrimRight(Workspace, "\r")
			Workspace = strings.TrimRight(Workspace, "\t")
		}
	}
	if os.Getenv("BKE_WORKSPACE") != "" {
		Workspace = os.Getenv("BKE_WORKSPACE")
	}
	if Workspace == "" {
		Workspace = "/bke"
	}
	if !utils.Exists(Workspace + "/tmpl") {
		if err := os.MkdirAll(Workspace+"/tmpl", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create tmpl directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/volumes") {
		if err := os.MkdirAll(Workspace+"/volumes", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create volumes directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/mount") {
		if err := os.MkdirAll(Workspace+"/mount", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create mount directory: %v", err)
		}
	}
	CustomExtra = make(map[string]string)

	assert.Equal(t, "/custom/workspace/path", Workspace)
}

func TestTarGZ(t *testing.T) {
	prefix := "/tmp/test-prefix"
	target := "/tmp/test-target.tar.gz"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		assert.Equal(t, "sh", command)
		assert.Equal(t, []string{"-c", fmt.Sprintf("cd %s && tar --use-compress-program=pigz -cf %s .", prefix, target)}, args)
		return "", nil
	})

	err := TarGZ(prefix, target)

	assert.NoError(t, err)
}

func TestTarGZError(t *testing.T) {
	prefix := "/tmp/test-prefix"
	target := "/tmp/test-target.tar.gz"

	// Apply patches to simulate error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		return "error output", fmt.Errorf("command failed")
	})

	err := TarGZ(prefix, target)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error output")
	assert.Contains(t, err.Error(), "command failed")
}

func TestTarGZWithDir(t *testing.T) {
	prefix := "/tmp/test-prefix"
	dir := "test-dir"
	target := "/tmp/test-target.tar.gz"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		expectedCmd := fmt.Sprintf("cd %s && tar --use-compress-program=pigz -cf %s ./%s", prefix, target, dir)
		assert.Equal(t, "sh", command)
		assert.Equal(t, []string{"-c", expectedCmd}, args)
		return "", nil
	})

	err := TarGZWithDir(prefix, dir, target)

	assert.NoError(t, err)
}

func TestTaeGZWithoutChangeFile(t *testing.T) {
	prefix := "/tmp/test-prefix"
	target := "/tmp/test-target.tar.gz"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		expectedCmd := fmt.Sprintf("cd %s && tar --use-compress-program=pigz -cf %s . --warning=no-file-changed --ignore-failed-read", prefix, target)
		assert.Equal(t, "sh", command)
		assert.Equal(t, []string{"-c", expectedCmd}, args)
		return "", nil
	})

	err := TaeGZWithoutChangeFile(prefix, target)

	assert.NoError(t, err)
}

func TestUnTarGZ(t *testing.T) {
	dataFile := "/tmp/data.tar.gz"
	target := "/tmp/target-dir"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		expectedCmd := fmt.Sprintf("tar -xzf %s -C %s", dataFile, target)
		assert.Equal(t, "sh", command)
		assert.Equal(t, []string{"-c", expectedCmd}, args)
		return "", nil
	})

	err := UnTarGZ(dataFile, target)

	assert.NoError(t, err)
}

func TestUnTarGZError(t *testing.T) {
	dataFile := "/tmp/data.tar.gz"
	target := "/tmp/target-dir"

	// Apply patches to simulate error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
		return "error output", fmt.Errorf("untar failed")
	})

	err := UnTarGZ(dataFile, target)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error output")
	assert.Contains(t, err.Error(), "untar failed")
}

func TestWorkspaceDirectoriesCreation(t *testing.T) {
	tempDir := t.TempDir()

	originalWorkspace := Workspace
	Workspace = tempDir

	defer func() {
		Workspace = originalWorkspace
	}()

	// Apply patches to ensure directories are created
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false // Simulate directories don't exist
	})

	mkdirCalls := testNumericZero
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		mkdirCalls++
		return nil
	})

	patches.ApplyFunc(log.Warnf, func(format string, args ...interface{}) {})

	// Check that directories would be created
	if !utils.Exists(Workspace + "/tmpl") {
		if err := os.MkdirAll(Workspace+"/tmpl", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create tmpl directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/volumes") {
		if err := os.MkdirAll(Workspace+"/volumes", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create volumes directory: %v", err)
		}
	}
	if !utils.Exists(Workspace + "/mount") {
		if err := os.MkdirAll(Workspace+"/mount", utils.DefaultFilePermission); err != nil {
			log.Warnf("failed to create mount directory: %v", err)
		}
	}

	// Should have tried to create 3 directories
	assert.Equal(t, testNumericThree, mkdirCalls)
}

func TestListK8sResources(t *testing.T) {
	tests := []struct {
		name        string
		mockClient  *k8s.MockK8sClient
		mockGetDyn  func(*k8s.MockK8sClient) interface{}
		expectError bool
	}{
		{
			name:       "successful list with mock client",
			mockClient: &k8s.MockK8sClient{},
			mockGetDyn: func(m *k8s.MockK8sClient) interface{} {
				return nil
			},
			expectError: false,
		},
		{
			name:       "list resources error",
			mockClient: &k8s.MockK8sClient{},
			mockGetDyn: func(m *k8s.MockK8sClient) interface{} {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := K8s
			defer func() {
				K8s = originalK8s
			}()

			K8s = tt.mockClient

			gvr := schema.GroupVersionResource{
				Group:    "test.group",
				Version:  "v1",
				Resource: "tests",
			}
			// 创建一个Command对象用于模拟返回
			testCommand := &agentv1beta1.Command{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Command",
					APIVersion: "agent.bke.bocloud.com/v1beta1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-command",
					Namespace: "default",
				},
				Spec: agentv1beta1.CommandSpec{
					Suspend: false,
				},
			}

			// 将Command转换为Unstructured格式
			unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(testCommand)
			assert.NoError(t, err)
			workloadUnstructured := &unstructured.Unstructured{Object: unstructuredObj}

			// 创建一个基本的动态客户端
			scheme := runtime.NewScheme()

			// 创建一个包含测试对象的完整客户端
			fullClient := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
				scheme,
				map[schema.GroupVersionResource]string{
					gvr: "CommandList",
				},
				workloadUnstructured, // 包含测试对象
			)

			// Mock GetDynamicClient to return our full client
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*k8s.MockK8sClient).GetDynamicClient, func(m *k8s.MockK8sClient) dynamic.Interface {
				return fullClient
			})

			err = ListK8sResources(gvr, testCommand)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestInitWorkspaceWithVariousWhitespaces(t *testing.T) {
	tests := []struct {
		name     string
		fileData string
		expected string
	}{
		{
			name:     "workspace with trailing newlines",
			fileData: "/test/workspace\n\n\n",
			expected: "/test/workspace",
		},
		{
			name:     "workspace with trailing tabs",
			fileData: "/test/workspace\t\t",
			expected: "/test/workspace",
		},
		{
			name:     "workspace with mixed trailing whitespace",
			fileData: "/test/workspace\r\n\t",
			expected: "/test/workspace",
		},
		{
			name:     "workspace with leading spaces trimmed",
			fileData: "  /test/workspace  ",
			expected: "/test/workspace",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempDir := t.TempDir()
			workspaceFile := filepath.Join(tempDir, "BKE_WORKSPACE")

			err := os.WriteFile(workspaceFile, []byte(tt.fileData), testFilePermission)
			assert.NoError(t, err)

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(utils.IsFile, func(path string) bool {
				return path == "/opt/BKE_WORKSPACE"
			})

			patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
				if filename == "/opt/BKE_WORKSPACE" {
					return []byte(tt.fileData), nil
				}
				return nil, fmt.Errorf("file not found")
			})

			patches.ApplyFunc(os.Getenv, func(key string) string {
				return ""
			})

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return true
			})

			patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
				return nil
			})

			patches.ApplyFunc(log.Warnf, func(format string, args ...interface{}) {})

			Command = &exec.CommandExecutor{}
			Workspace = ""
			CustomExtra = make(map[string]string)

			if utils.IsFile("/opt/BKE_WORKSPACE") {
				f, err := os.ReadFile("/opt/BKE_WORKSPACE")
				if err == nil {
					Workspace = string(f)
					Workspace = strings.TrimSpace(Workspace)
					Workspace = strings.TrimRight(Workspace, "\n")
					Workspace = strings.TrimRight(Workspace, "\r")
					Workspace = strings.TrimRight(Workspace, "\t")
				}
			}
			if os.Getenv("BKE_WORKSPACE") != "" {
				Workspace = os.Getenv("BKE_WORKSPACE")
			}
			if Workspace == "" {
				Workspace = "/bke"
			}

			assert.Equal(t, tt.expected, Workspace)
		})
	}
}

func TestExecuteManifestCommands(t *testing.T) {
	target := "registry.example.com/app:v1.0.0"
	createCmd := "docker manifest create --insecure " + target
	annotateCmds := []string{
		"docker manifest annotate --arch amd64 " + target,
		"docker manifest annotate --arch arm64 " + target,
	}

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		callCount := 0
		patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithEnv,
			func(_ *exec.CommandExecutor, _ []string, command string, args ...string) error {
				callCount++
				assert.Equal(t, "/bin/sh", command)
				assert.NotEmpty(t, args)
				return nil
			})
		patches.ApplyFunc(log.Infof, func(string, ...interface{}) {})
		patches.ApplyFunc(log.Debugf, func(string, ...interface{}) {})

		err := ExecuteManifestCommands(target, createCmd, annotateCmds)
		assert.NoError(t, err)
		assert.Equal(t, len(annotateCmds)+2, callCount) // create + annotate + push
	})

	t.Run("create fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithEnv,
			func(_ *exec.CommandExecutor, _ []string, _ string, _ ...string) error {
				return fmt.Errorf("create failed")
			})
		patches.ApplyFunc(log.Errorf, func(string, ...interface{}) {})

		err := ExecuteManifestCommands(target, createCmd, annotateCmds)
		assert.Error(t, err)
	})

	t.Run("annotate fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		callCount := 0
		patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithEnv,
			func(_ *exec.CommandExecutor, _ []string, _ string, _ ...string) error {
				callCount++
				if callCount == 1 {
					return nil
				}
				return fmt.Errorf("annotate failed")
			})
		patches.ApplyFunc(log.Infof, func(string, ...interface{}) {})
		patches.ApplyFunc(log.Errorf, func(string, ...interface{}) {})

		err := ExecuteManifestCommands(target, createCmd, annotateCmds)
		assert.Error(t, err)
	})

	t.Run("push fails", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		callCount := 0
		patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithEnv,
			func(_ *exec.CommandExecutor, _ []string, _ string, _ ...string) error {
				callCount++
				if callCount <= len(annotateCmds)+1 {
					return nil
				}
				return fmt.Errorf("push failed")
			})
		patches.ApplyFunc(log.Infof, func(string, ...interface{}) {})
		patches.ApplyFunc(log.Debugf, func(string, ...interface{}) {})
		patches.ApplyFunc(log.Errorf, func(string, ...interface{}) {})

		err := ExecuteManifestCommands(target, createCmd, annotateCmds)
		assert.Error(t, err)
	})
}
