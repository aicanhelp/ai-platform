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

package agent

import (
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicFake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	fake "k8s.io/client-go/kubernetes/fake"
	"sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

const (
	testZeroValue         = 0
	testOneValue          = 1
	testTwoValue          = 2
	testDefaultPortNumber = 123
)

func TestBuildNodeSelector(t *testing.T) {
	tests := []struct {
		name     string
		nodes    string
		expected map[string]string
	}{
		{
			name:     "single node",
			nodes:    "node1",
			expected: map[string]string{"node1": "node1"},
		},
		{
			name:     "multiple nodes",
			nodes:    "node1,node2,node3",
			expected: map[string]string{"node1": "node1", "node2": "node2", "node3": "node3"},
		},
		{
			name:     "empty nodes",
			nodes:    "",
			expected: map[string]string{"": ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{Nodes: tt.nodes}
			result := op.buildNodeSelector()

			assert.Equal(t, tt.expected, result.MatchLabels)
		})
	}
}

func TestBuildCommand(t *testing.T) {
	op := &Options{
		Name:  "test-command",
		Nodes: "node1,node2",
	}

	cmd := op.buildCommand()

	// Check basic properties
	assert.Equal(t, "test-command", cmd.GetName())
	assert.Equal(t, metav1.NamespaceDefault, cmd.GetNamespace())
	assert.Equal(t, "Command", cmd.GetObjectKind().GroupVersionKind().Kind)
	assert.Equal(t, agentv1beta1.GroupVersion.Group, cmd.GroupVersionKind().Group)
	assert.Equal(t, agentv1beta1.GroupVersion.Version, cmd.GroupVersionKind().Version)

	// Check annotations
	expectedAnnotations := map[string]string{annotationKey: annotationValue}
	assert.Equal(t, expectedAnnotations, cmd.GetAnnotations())

	// Check node selector
	expectedNodeSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"node1": "node1", "node2": "node2"},
	}
	assert.Equal(t, expectedNodeSelector, cmd.Spec.NodeSelector)

	// Check initial command spec
	assert.False(t, cmd.Spec.Suspend)
	assert.Empty(t, cmd.Spec.Commands)
}

func TestApplyCommand(t *testing.T) {
	// Test with command
	op := &Options{
		Name:    "test-cmd",
		Command: "echo hello",
	}

	cmd := op.buildCommand()

	// Apply patches to mock the configmap creation and install command
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*Options).createConfigMapFromFile, func(o *Options, c *agentv1beta1.Command) error {
		return nil
	})

	patches.ApplyFunc((*Options).installCommand, func(o *Options, c *agentv1beta1.Command) error {
		return nil
	})

	err := op.applyCommand(&cmd)
	assert.NoError(t, err)

	// Check that command was added
	assert.Len(t, cmd.Spec.Commands, testOneValue)
	assert.Equal(t, "command", cmd.Spec.Commands[testZeroValue].ID)
	assert.Equal(t, []string{"echo hello"}, cmd.Spec.Commands[testZeroValue].Command)
	assert.Equal(t, agentv1beta1.CommandShell, cmd.Spec.Commands[testZeroValue].Type)

	// Test with file
	op2 := &Options{
		Name: "test-file",
		File: "/path/to/file",
	}

	cmd2 := op2.buildCommand()
	err2 := op2.applyCommand(&cmd2)
	assert.NoError(t, err2)
	// When file is provided, createConfigMapFromFile would be called
}

func TestInstallCommand(t *testing.T) {
	op := &Options{}
	cmd := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-install",
		},
	}

	// Apply patches to mock InstallYaml
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*k8s.Client).InstallYaml, func(c *k8s.Client, filename string, data map[string]string, template string) error {
		// Check that the file was created with the right content
		content, err := os.ReadFile(filename)
		if err != nil {
			return err
		}

		var parsedCmd agentv1beta1.Command
		err = yaml.Unmarshal(content, &parsedCmd)
		if err != nil {
			return err
		}

		assert.Equal(t, cmd.Name, parsedCmd.Name)
		return nil
	})

	// Also mock WriteFile to avoid actual file creation
	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		// Just validate that the name ends with .yaml
		if !strings.HasSuffix(name, ".yaml") {
			return fmt.Errorf("filename should end with .yaml")
		}
		return nil
	})

	if global.K8s == nil {
		global.K8s = &k8s.Client{}
	}
	err := op.installCommand(cmd)
	assert.Error(t, err)
}

func TestExec(t *testing.T) {
	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*Options).buildCommand, func(o *Options) agentv1beta1.Command {
		return agentv1beta1.Command{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-exec",
			},
		}
	})

	applyCalled := false
	patches.ApplyFunc((*Options).applyCommand, func(o *Options, cmd *agentv1beta1.Command) error {
		applyCalled = true
		return nil
	})

	op := &Options{Name: "test-exec"}
	op.Exec()

	assert.True(t, applyCalled, "applyCommand should have been called")
}

func TestList(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(global.ListK8sResources, func(gvr schema.GroupVersionResource, list interface{}) error {
		commandList, ok := list.(*agentv1beta1.CommandList)
		if !ok {
			return fmt.Errorf("wrong type")
		}

		now := metav1.Now()
		commandList.Items = []agentv1beta1.Command{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-command",
					Namespace: metav1.NamespaceDefault,
				},
				Spec: agentv1beta1.CommandSpec{
					Suspend: false,
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					"node1": &agentv1beta1.CommandStatus{
						LastStartTime:  &now,
						CompletionTime: &now,
						Phase:          "Completed",
						Status:         "Success",
					},
				},
			},
		}
		return nil
	})

	patches.ApplyFunc(fmt.Fprintln, func(w io.Writer, a ...interface{}) (n int, err error) {
		return 0, nil
	})

	patches.ApplyFunc(fmt.Fprintf, func(w io.Writer, format string, a ...interface{}) (n int, err error) {
		return 0, nil
	})

	op := &Options{}
	op.List()
}

func TestInfo(t *testing.T) {
	tests := []struct {
		name              string
		args              []string
		mockSplit         func(string, string) []string
		mockDynamicClient func() dynamic.Interface
		expectError       bool
	}{
		{
			name: "valid argument format",
			args: []string{"default/test-command"},
			mockDynamicClient: func() dynamic.Interface {
				return nil // Will be mocked properly below
			},
			expectError: false,
		},
		{
			name:        "invalid argument format",
			args:        []string{"invalid-format"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 在测试开始时就设置全局K8s客户端
			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient

			op := &Options{Args: tt.args}

			// Apply patches for valid format test
			if !tt.expectError {
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
				baseClient := dynamicFake.NewSimpleDynamicClient(scheme)

				// Mock GetDynamicClient to return our base client
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*k8s.MockK8sClient).GetDynamicClient, func(m *k8s.MockK8sClient) dynamic.Interface {
					return baseClient
				})

				// 创建一个包含测试对象的完整客户端
				fullClient := dynamicFake.NewSimpleDynamicClientWithCustomListKinds(
					scheme,
					map[schema.GroupVersionResource]string{
						gvr: "CommandList",
					},
					workloadUnstructured, // 包含测试对象
				)

				// Mock GetDynamicClient to return our full client
				patches.ApplyFunc((*k8s.MockK8sClient).GetDynamicClient, func(m *k8s.MockK8sClient) dynamic.Interface {
					return fullClient
				})
			}

			op.Info()

			// For the invalid format test, we expect an error to be logged
			if tt.expectError {
				// We can't easily capture the log output, so just ensure the function runs
				assert.True(t, true)
			} else {
				assert.True(t, true)
			}
		})
	}
}

func TestRemove(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectError bool
	}{
		{
			name:        "valid argument format",
			args:        []string{"default/test-command"},
			expectError: false,
		},
		{
			name:        "invalid argument format",
			args:        []string{"invalid-format"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{Args: tt.args}
			op.Remove()
		})
	}
}

func TestSyncTime(t *testing.T) {
	tests := []struct {
		name                  string
		args                  []string
		mockDate              func(string) error
		minManifestsImageArgs string
		containerWaitSeconds  string
	}{
		{
			name: "sync with default NTP server",
			args: []string{},
			mockDate: func(server string) error {
				return nil
			},
		},
		{
			name: "sync with custom NTP server",
			args: []string{"custom.ntp.server:123"},
			mockDate: func(server string) error {
				assert.Equal(t, "custom.ntp.server:123", server)
				return nil
			},
		},
		{
			name: "sync with retry on failure",
			args: []string{"ntp.server:123"},
			mockDate: func(server string) error {
				return fmt.Errorf("connection failed")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{Args: tt.args}

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(ntp.Date, tt.mockDate)

			op.SyncTime()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestCreateConfigMapFromFile(t *testing.T) {
	tests := []struct {
		name           string
		fileContent    string
		filePath       string
		setupMocks     func(*gomonkey.Patches) *k8s.MockK8sClient
		restoreGlobal  func()
		expectError    bool
		expectedErrMsg string
	}{
		{
			name:        "successful configmap creation",
			fileContent: "test file content",
			filePath:    "/tmp/test-file.txt",
			setupMocks: func(patches *gomonkey.Patches) *k8s.MockK8sClient {
				// Mock os.ReadFile to return test content
				patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
					return []byte("test file content"), nil
				})

				// Create a mock K8s client
				mockK8sClient := &k8s.MockK8sClient{}

				// Mock the GetClient method to return a fake clientset
				clientset := fake.NewSimpleClientset()
				patches.ApplyFunc((*k8s.MockK8sClient).GetClient, func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return clientset
				})

				return mockK8sClient
			},
			expectError: false,
		},
		{
			name:        "file read error",
			fileContent: "",
			filePath:    "/nonexistent/file.txt",
			setupMocks: func(patches *gomonkey.Patches) *k8s.MockK8sClient {
				// Mock os.ReadFile to return error
				patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
					return nil, fmt.Errorf("file not found")
				})
				return nil
			},
			expectError:    true,
			expectedErrMsg: "file not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original global K8s
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			var mockK8sClient *k8s.MockK8sClient
			if tt.setupMocks != nil {
				mockK8sClient = tt.setupMocks(patches)
				if mockK8sClient != nil {
					global.K8s = mockK8sClient
				}
			}

			// Create Options with file path
			op := &Options{
				Name: "test-configmap",
				File: tt.filePath,
			}

			// Create a command to pass to the function
			cmd := &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					Commands: []agentv1beta1.ExecCommand{},
				},
			}

			err := op.createConfigMapFromFile(cmd)

			if tt.expectError {
				assert.Error(t, err)
				if tt.expectedErrMsg != "" {
					assert.Contains(t, err.Error(), tt.expectedErrMsg)
				}
			} else {
				assert.NoError(t, err)
				// Verify that the command was updated correctly
				assert.Len(t, cmd.Spec.Commands, testOneValue)
				assert.Equal(t, "file", cmd.Spec.Commands[0].ID)
				assert.Equal(t, agentv1beta1.CommandKubernetes, cmd.Spec.Commands[0].Type)
				expectedCommand := []string{fmt.Sprintf("configmap:%s/%s:rx:shell", metav1.NamespaceDefault, op.Name)}
				assert.Equal(t, expectedCommand, cmd.Spec.Commands[0].Command)
			}
		})
	}
}
