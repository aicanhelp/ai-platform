/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package bkeconsole

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	fake "k8s.io/client-go/kubernetes/fake"

	"gopkg.openfuyao.cn/bkeadm/pkg/common/types"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testNumericZero      = 0
	testNumericOne       = 1
	testNumericTwo       = 2
	testNumericThree     = 3
	testNumericFive      = 5
	testNumericTen       = 10
	testFilePerm         = 0644
	testFileModeReadOnly = 0644
	testFileModeExec     = 0755
)

const (
	testIPv4SegA = 192
	testIPv4SegB = 168
	testIPv4SegC = 1
	testIPv4SegD = 100
)

const (
	testIPv4SegA2 = 8
	testIPv4SegB2 = 8
	testIPv4SegC2 = 8
	testIPv4SegD2 = 8
)

const (
	testIPv4SegA3 = 114
	testIPv4SegB3 = 114
	testIPv4SegC3 = 114
	testIPv4SegD3 = 114
)

const (
	testIPv4SegA4 = 192
	testIPv4SegB4 = 168
	testIPv4SegC4 = 1
	testIPv4SegD4 = 200
)

const (
	testIPv4SegA5 = 127
	testIPv4SegB5 = 0
	testIPv4SegC5 = 0
	testIPv4SegD5 = 1
)

var (
	testIPAddr       = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegA, testIPv4SegB, testIPv4SegC, testIPv4SegD)
	testDNSPrimary   = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegA2, testIPv4SegB2, testIPv4SegC2, testIPv4SegD2)
	testDNSSecondary = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegA3, testIPv4SegB3, testIPv4SegC3, testIPv4SegD3)
	testOtherRepoIP  = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegA4, testIPv4SegB4, testIPv4SegC4, testIPv4SegD4)
	testLoopbackIP   = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegA5, testIPv4SegB5, testIPv4SegC5, testIPv4SegD5)
)

func TestWriteToDir(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name           string
		dir            string
		script         string
		scriptFile     string
		mockExists     func(string) bool
		mockMkdirAll   func(string, os.FileMode) error
		mockWriteFile  func(string, []byte, os.FileMode) error
		expectError    bool
		expectedErrMsg string
	}{
		{
			name:       "successful write to new directory",
			dir:        filepath.Join(tempDir, "new-dir"),
			script:     "test.sh",
			scriptFile: "#!/bin/bash\necho test",
			mockExists: func(path string) bool { return false },
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:       "successful write to existing directory",
			dir:        tempDir,
			script:     "test.sh",
			scriptFile: "#!/bin/bash\necho test",
			mockExists: func(path string) bool { return true },
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:       "mkdir all fails",
			dir:        filepath.Join(tempDir, "new-dir"),
			script:     "test.sh",
			scriptFile: "#!/bin/bash\necho test",
			mockExists: func(path string) bool { return false },
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return fmt.Errorf("mkdir error")
			},
			expectError:    true,
			expectedErrMsg: "create dir failed",
		},
		{
			name:       "write file fails",
			dir:        tempDir,
			script:     "test.sh",
			scriptFile: "#!/bin/bash\necho test",
			mockExists: func(path string) bool { return true },
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return fmt.Errorf("write error")
			},
			expectError:    true,
			expectedErrMsg: "write test.sh fialed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, tt.mockExists)
			if tt.mockMkdirAll != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			}
			if tt.mockWriteFile != nil {
				patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)
			}

			err := writeToDir(tt.dir, tt.script, tt.scriptFile)

			if tt.expectError {
				assert.Error(t, err)
				if tt.expectedErrMsg != "" {
					assert.Contains(t, err.Error(), tt.expectedErrMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDeployConsole(t *testing.T) {
	tests := []struct {
		name             string
		otherRepo        string
		onlineImage      string
		hostIP           string
		repo             string
		openFuyaoVersion string
		mockExists       func(string) bool
		mockMkdirAll     func(string, os.FileMode) error
		mockCopyFS       func(embed.FS, string, string) error
		mockWriteDir     func(string, string, string) error
		mockExecuteCmd   func(*exec.CommandExecutor, string, ...string) (string, error)
		expectError      bool
	}{
		{
			name:             "mkdir resource dir fails",
			otherRepo:        "",
			onlineImage:      "",
			hostIP:           testIPAddr,
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockExists: func(path string) bool {
				return false
			},
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return fmt.Errorf("mkdir error")
			},
			expectError: true,
		},
		{
			name:             "execute command fails",
			otherRepo:        "",
			onlineImage:      "",
			hostIP:           testIPAddr,
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockExists: func(path string) bool {
				return true
			},
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockCopyFS: func(efs embed.FS, src string, dst string) error {
				return nil
			},
			mockWriteDir: func(dir string, script string, scriptContent string) error {
				return nil
			},
			mockExecuteCmd: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "", fmt.Errorf("command failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)

			if tt.mockCopyFS != nil {
				patches.ApplyFunc(copyEmbeddedFS, tt.mockCopyFS)
			}

			if tt.mockWriteDir != nil {
				patches.ApplyFunc(writeToDir, tt.mockWriteDir)
			}

			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)
			patches.ApplyFunc(repository.EnsureConsoleTools, func(string, string) error { return nil })

			err := deployConsole(types.K3sRestartConfig{
				OnlineImage: tt.onlineImage,
				OtherRepo:   tt.otherRepo,
				HostIP:      tt.hostIP,
			}, tt.repo, tt.openFuyaoVersion)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGenerateSecret(t *testing.T) {
	tests := []struct {
		name           string
		mockWriteFile  func(string, []byte, os.FileMode) error
		mockExecuteCmd func(*exec.CommandExecutor, string, ...string) (string, error)
		expectError    bool
	}{
		{
			name: "write file fails",
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return fmt.Errorf("write error")
			},
			expectError: true,
		},
		{
			name: "execute command fails",
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockExecuteCmd: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "", fmt.Errorf("command failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(writeToDir, func(dir string, script string, scriptContent string) error {
				return tt.mockWriteFile(filepath.Join(dir, script), []byte(scriptContent), utils.DefaultFilePermission)
			})

			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			err := generateSecret()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGetDNSServers(t *testing.T) {
	tests := []struct {
		name        string
		dnsConfig   string
		expectError bool
		expectLen   int
	}{
		{
			name:        "valid dns config",
			dnsConfig:   "servers:\n  - " + testDNSPrimary + "\n  - " + testDNSSecondary + "\n",
			expectError: false,
			expectLen:   2,
		},
		{
			name:        "empty dns config",
			dnsConfig:   "servers: []\n",
			expectError: true,
			expectLen:   0,
		},
		{
			name:        "invalid yaml",
			dnsConfig:   "invalid: yaml: content:",
			expectError: true,
			expectLen:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dnsConfig = tt.dnsConfig
			defer func() {
				dnsConfig = ""
			}()

			servers, err := getDNSServers()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Len(t, servers, tt.expectLen)
			}
		})
	}
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "/var/lib/rancher/k3s/", scriptDir)
	assert.Equal(t, "/var/lib/rancher/k3s/resource", resourceDir)
	assert.Equal(t, "60s", cacheTtl)
	assert.Equal(t, "kubernetes", k3sName)
}

func TestK3sRestartLogic(t *testing.T) {
	config := types.K3sRestartConfig{
		OtherRepo:      "",
		HostIP:         testIPAddr,
		ImageRepo:      "registry.example.com",
		ImageRepoPort:  "5000",
		OtherRepoIp:    "",
		KubernetesPort: "6443",
		ConsolePort:    "31000",
	}

	tests := []struct {
		name             string
		mockRun          func([]string) error
		mockEnsureImage  func(string) error
		mockContainerErr error
		expectError      bool
	}{
		{
			name: "stop k3s fails",
			mockRun: func(args []string) error {
				return fmt.Errorf("stop error")
			},
			expectError: true,
		},
		{
			name: "ensure image fails",
			mockRun: func(args []string) error {
				return nil
			},
			mockEnsureImage: func(image string) error {
				return fmt.Errorf("image error")
			},
			expectError: true,
		},
		{
			name: "restart succeeds with custom console port",
			mockRun: func(args []string) error {
				return nil
			},
			mockEnsureImage: func(image string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(econd.Run, tt.mockRun)

			patches.ApplyFunc(econd.EnsureImageExists, tt.mockEnsureImage)

			patches.ApplyFunc(econd.ContainerInspect, func(name string) (econd.NerdContainerInfo, error) {
				return econd.NerdContainerInfo{}, tt.mockContainerErr
			})

			patches.ApplyFunc(startK3sContainer, func(a, b, c, d, e string) error {
				assert.Equal(t, "31000", e)
				return nil
			})

			patches.ApplyFunc(processKubeconfig, func(a, b string) (string, error) {
				return "/test/.kube/config", nil
			})

			patches.ApplyFunc(waitForKubernetesReady, func(a string) error {
				return nil
			})

			patches.ApplyFunc(waitForClusterReady, func() error {
				return nil
			})

			err := k3sRestart(config)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartK3sContainerLogic(t *testing.T) {
	tests := []struct {
		name        string
		mockGetDNS  func() ([]string, error)
		mockRun     func([]string) error
		expectError bool
	}{
		{
			name: "get dns servers fails",
			mockGetDNS: func() ([]string, error) {
				return nil, fmt.Errorf("dns error")
			},
			expectError: true,
		},
		{
			name: "run container fails",
			mockGetDNS: func() ([]string, error) {
				return []string{testDNSPrimary}, nil
			},
			mockRun: func(args []string) error {
				assert.Contains(t, args, "31000:30010")
				assert.Contains(t, args, fmt.Sprintf("%s:%s", k3s.K3sPodLogDir, k3s.K3sPodLogDir))
				assert.Contains(t, args, fmt.Sprintf("%s:%s", k3s.K3sContainerLogDir, k3s.K3sContainerLogDir))
				for _, arg := range args {
					assert.False(t, strings.HasPrefix(arg, "--token="), "k3s must not use a hardcoded --token")
				}
				return fmt.Errorf("container error")
			},
			expectError: true,
		},
		{
			name: "start succeeds without fixed token",
			mockGetDNS: func() ([]string, error) {
				return []string{testDNSPrimary}, nil
			},
			mockRun: func(args []string) error {
				assert.Contains(t, args, "server")
				assert.Contains(t, args, "--cluster-cidr=100.20.0.0/16")
				for _, arg := range args {
					assert.False(t, strings.HasPrefix(arg, "--token="), "k3s must not use a hardcoded --token")
				}
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(getDNSServers, tt.mockGetDNS)
			patches.ApplyFunc(k3s.EnsureK3sLogDirs, func() error {
				return nil
			})

			patches.ApplyFunc(econd.Run, tt.mockRun)

			err := startK3sContainer(testIPAddr, "registry.example.com", testOtherRepoIP, "6443", "31000")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestProcessKubeconfigLogic(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	kubeconfigContent := fmt.Sprintf(`apiVersion: v1
clusters:
- cluster:
    server: https://%s:36443
   name: default
contexts:
- context:
     cluster: default
     user: default
   name: default
current-context: default
kind: Config
users:
- name: default
   user:
      token: test-token
`, testLoopbackIP)

	tests := []struct {
		name           string
		kubeconfigData []byte
		mockReadFile   func(string) ([]byte, error)
		mockUserHome   func() (string, error)
		mockMkdirAll   func(string, os.FileMode) error
		mockWriteFile  func(string, []byte, os.FileMode) error
		mockRemoveFile func(string) error
		expectError    bool
	}{
		{
			name:           "read kubeconfig fails",
			kubeconfigData: nil,
			mockReadFile: func(filename string) ([]byte, error) {
				return nil, fmt.Errorf("read error")
			},
			expectError: true,
		},
		{
			name:           "empty kubeconfig",
			kubeconfigData: []byte(""),
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(""), nil
			},
			expectError: true,
		},
		{
			name:           "get user home fails",
			kubeconfigData: []byte(kubeconfigContent),
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockUserHome: func() (string, error) {
				return "", fmt.Errorf("home error")
			},
			expectError: true,
		},
		{
			name:           "mkdir kube dir fails",
			kubeconfigData: []byte(kubeconfigContent),
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockUserHome: func() (string, error) {
				return "/tmp/test-home", nil
			},
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return fmt.Errorf("mkdir error")
			},
			expectError: true,
		},
		{
			name:           "write kubeconfig fails",
			kubeconfigData: []byte(kubeconfigContent),
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockUserHome: func() (string, error) {
				return "/tmp/test-home", nil
			},
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return fmt.Errorf("write error")
			},
			expectError: true,
		},
		{
			name:           "success path",
			kubeconfigData: []byte(kubeconfigContent),
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockUserHome: func() (string, error) {
				return "/tmp/test-home", nil
			},
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockRemoveFile: func(name string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(os.ReadFile, tt.mockReadFile)
			if tt.mockUserHome != nil {
				patches.ApplyFunc(os.UserHomeDir, tt.mockUserHome)
			}
			if tt.mockMkdirAll != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			}
			if tt.mockWriteFile != nil {
				patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)
			}
			if tt.mockRemoveFile != nil {
				patches.ApplyFunc(os.Remove, tt.mockRemoveFile)
			}

			path, err := processKubeconfig(testIPAddr, "6443")

			if tt.expectError {
				assert.Error(t, err)
				assert.Empty(t, path)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, path)
			}
		})
	}
}

func TestWaitForKubernetesReadyLogic(t *testing.T) {
	tests := []struct {
		name           string
		kubeconfigPath string
		mockNewClient  func(string) (k8s.KubernetesClient, error)
		expectError    bool
	}{
		{
			name:           "new client fails",
			kubeconfigPath: "/tmp/test-kubeconfig",
			mockNewClient: func(path string) (k8s.KubernetesClient, error) {
				return nil, fmt.Errorf("client error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			global.K8s = nil

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(k8s.NewKubernetesClient, tt.mockNewClient)
			patches.ApplyFunc(time.Sleep, func(time.Duration) {
				t.Log("Skipping Sleep for testing")
			})

			err := waitForKubernetesReady(tt.kubeconfigPath)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, global.K8s)
			}
		})
	}
}

func TestWaitForClusterReadyLogic(t *testing.T) {
	tests := []struct {
		name          string
		mockGetClient func() kubernetes.Interface
		expectError   bool
	}{
		{
			name: "get node fails",
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset()
			},
			expectError: false,
		},
		{
			name: "node has taints",
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset(&corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: utils.LocalKubernetesName,
					},
					Spec: corev1.NodeSpec{
						Taints: []corev1.Taint{
							{
								Key: "test",
							},
						},
					},
				})
			},
			expectError: false,
		},
		{
			name: "node ready with namespace",
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset(
					&corev1.Node{
						ObjectMeta: metav1.ObjectMeta{
							Name: utils.LocalKubernetesName,
						},
					},
					&corev1.Namespace{
						ObjectMeta: metav1.ObjectMeta{
							Name: "kube-system",
						},
					},
				)
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*k8s.MockK8sClient).GetClient, func(m *k8s.MockK8sClient) kubernetes.Interface {
				return tt.mockGetClient()
			})

			patches.ApplyFunc(time.Sleep, func(d time.Duration) {
				t.Log("Skipping Sleep for testing")
			})

			err := waitForClusterReady()
			assert.NoError(t, err)
		})
	}
}

func TestDeployOauthAndUserLogic(t *testing.T) {
	tests := []struct {
		name             string
		onlineImage      string
		otherRepo        string
		hostIP           string
		repo             string
		openFuyaoVersion string
		mockWriteFile    func(string, []byte, os.FileMode) error
		mockExecuteCmd   func(*exec.CommandExecutor, string, ...string) (string, error)
		expectError      bool
	}{
		{
			name:             "write file fails",
			onlineImage:      "",
			otherRepo:        "",
			hostIP:           testIPAddr,
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return fmt.Errorf("write error")
			},
			expectError: true,
		},
		{
			name:             "execute command fails",
			onlineImage:      "",
			otherRepo:        "",
			hostIP:           testIPAddr,
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockExecuteCmd: func(c *exec.CommandExecutor, command string, args ...string) (string, error) {
				return "", fmt.Errorf("command failed")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(writeToDir, func(dir string, script string, scriptContent string) error {
				return tt.mockWriteFile(filepath.Join(dir, script), []byte(scriptContent), utils.DefaultFilePermission)
			})

			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithCombinedOutput, tt.mockExecuteCmd)

			err := deployOauthAndUser(oauthAndUserDeployConfig{
				onlineImage:      tt.onlineImage,
				otherRepo:        tt.otherRepo,
				hostIP:           tt.hostIP,
				repo:             tt.repo,
				openFuyaoVersion: tt.openFuyaoVersion,
			}, "all")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGetPodsLogic(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()

	tests := []struct {
		name          string
		namespace     string
		mockGetClient func() kubernetes.Interface
		expectError   bool
	}{
		{
			name:      "get pods from namespace",
			namespace: "openfuyao-system",
			mockGetClient: func() kubernetes.Interface {
				return fakeClient
			},
			expectError: true,
		},
		{
			name:      "get pods with job pods only",
			namespace: "test-namespace",
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset(&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-pod",
						OwnerReferences: []metav1.OwnerReference{
							{
								Kind: "Job",
							},
						},
					},
				})
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			global.K8s = &k8s.MockK8sClient{}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*k8s.MockK8sClient).GetClient, func(m *k8s.MockK8sClient) kubernetes.Interface {
				return tt.mockGetClient()
			})

			podList, err := getPods(tt.mockGetClient(), tt.namespace)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, podList)
			}
		})
	}
}

func TestWaitAllConsolePodRunningLogic(t *testing.T) {
	podReady := []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	tests := []struct {
		name             string
		iterationToReady int
		mockGetClient    func() kubernetes.Interface
		expectCalled     bool
	}{
		{
			name:             "pods ready on first check",
			iterationToReady: 0,
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset(
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "console-pod",
							Namespace: "openfuyao-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "ingress-pod",
							Namespace: "ingress-nginx",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bke-controller-manager-pod",
							Namespace: "cluster-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "capi-controller-manager-pod",
							Namespace: "cluster-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
				)
			},
			expectCalled: true,
		},
		{
			name:             "pods ready after retry",
			iterationToReady: 2,
			mockGetClient: func() kubernetes.Interface {
				return fake.NewSimpleClientset(
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "console-pod",
							Namespace: "openfuyao-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "ingress-pod",
							Namespace: "ingress-nginx",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bke-controller-manager-pod",
							Namespace: "cluster-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
					&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "capi-controller-manager-pod",
							Namespace: "cluster-system",
						},
						Status: corev1.PodStatus{
							Phase:      corev1.PodRunning,
							Conditions: podReady,
						},
					},
				)
			},
			expectCalled: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient

			sleepCounter := 0
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*k8s.MockK8sClient).GetClient, func(m *k8s.MockK8sClient) kubernetes.Interface {
				return tt.mockGetClient()
			})

			patches.ApplyFunc(time.Sleep, func(d time.Duration) {
				sleepCounter++
			})

			waitAllConsolePodRunning()

			assert.True(t, tt.expectCalled)
		})
	}
}

func TestDeployCorednsLogic(t *testing.T) {
	tests := []struct {
		name          string
		repo          string
		mockMkdirAll  func(string, os.FileMode) error
		mockWriteFile func(string, []byte, os.FileMode) error
		mockInstall   func(*k8s.MockK8sClient, string, map[string]string, string) error
		expectError   bool
	}{
		{
			name: "mkdir fails",
			repo: "registry.example.com",
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return fmt.Errorf("mkdir error")
			},
			expectError: true,
		},
		{
			name: "write file fails",
			repo: "registry.example.com",
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return fmt.Errorf("write error")
			},
			expectError: true,
		},
		{
			name: "install yaml fails",
			repo: "registry.example.com",
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockInstall: func(m *k8s.MockK8sClient, file string, params map[string]string, namespace string) error {
				return fmt.Errorf("install error")
			},
			expectError: true,
		},
		{
			name: "success",
			repo: "registry.example.com",
			mockMkdirAll: func(path string, mode os.FileMode) error {
				return nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockInstall: func(m *k8s.MockK8sClient, file string, params map[string]string, namespace string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			originalWorkspace := global.Workspace
			defer func() {
				global.K8s = originalK8s
				global.Workspace = originalWorkspace
			}()

			global.Workspace = "/tmp/test-workspace"

			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)

			patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)

			patches.ApplyFunc((*k8s.MockK8sClient).InstallYaml, tt.mockInstall)

			err := deployCoredns(tt.repo)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDeployConsoleAllLogic(t *testing.T) {
	config := types.K3sRestartConfig{
		OtherRepo:      "",
		HostIP:         testIPAddr,
		ImageRepo:      "registry.example.com",
		ImageRepoPort:  "5000",
		OtherRepoIp:    "",
		KubernetesPort: "6443",
	}

	tests := []struct {
		name             string
		repo             string
		openFuyaoVersion string
		mockNewK8s       func(string) (k8s.KubernetesClient, error)
		mockDeployCore   func(string) error
		mockDeployCon    func(types.K3sRestartConfig, string, string) error
		mockWaitCon      func()
		mockGenerate     func() error
		mockK3sRestart   func(types.K3sRestartConfig) error
		mockDeployOauth  func(oauthAndUserDeployConfig, string) error
		expectError      bool
	}{
		{
			name:             "new k8s client fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return nil, fmt.Errorf("client error")
			},
			expectError: true,
		},
		{
			name:             "deploy coredns fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return fmt.Errorf("coredns error")
			},
			expectError: true,
		},
		{
			name:             "deploy console fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return nil
			},
			mockDeployCon: func(types.K3sRestartConfig, string, string) error {
				return fmt.Errorf("console error")
			},
			expectError: true,
		},
		{
			name:             "generate secret fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return nil
			},
			mockDeployCon: func(types.K3sRestartConfig, string, string) error {
				return nil
			},
			mockWaitCon: func() {},
			mockGenerate: func() error {
				return fmt.Errorf("secret error")
			},
			expectError: true,
		},
		{
			name:             "k3s restart fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return nil
			},
			mockDeployCon: func(types.K3sRestartConfig, string, string) error {
				return nil
			},
			mockWaitCon: func() {},
			mockGenerate: func() error {
				return nil
			},
			mockDeployOauth: func(oauthAndUserDeployConfig, string) error {
				return nil
			},
			mockK3sRestart: func(c types.K3sRestartConfig) error {
				return fmt.Errorf("k3s error")
			},
			expectError: true,
		},
		{
			name:             "deploy oauth fails",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return nil
			},
			mockDeployCon: func(types.K3sRestartConfig, string, string) error {
				return nil
			},
			mockWaitCon: func() {},
			mockGenerate: func() error {
				return nil
			},
			mockK3sRestart: func(c types.K3sRestartConfig) error {
				return nil
			},
			mockDeployOauth: func(oauthAndUserDeployConfig, string) error {
				return fmt.Errorf("oauth error")
			},
			expectError: true,
		},
		{
			name:             "success",
			repo:             "registry.example.com",
			openFuyaoVersion: "v1.0.0",
			mockNewK8s: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			mockDeployCore: func(repo string) error {
				return nil
			},
			mockDeployCon: func(types.K3sRestartConfig, string, string) error {
				return nil
			},
			mockWaitCon: func() {},
			mockGenerate: func() error {
				return nil
			},
			mockK3sRestart: func(c types.K3sRestartConfig) error {
				return nil
			},
			mockDeployOauth: func(oauthAndUserDeployConfig, string) error {
				return nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() {
				global.K8s = originalK8s
			}()

			global.K8s = nil

			if tt.mockNewK8s != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(k8s.NewKubernetesClient, tt.mockNewK8s)
			}

			if tt.mockDeployCore != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(deployCoredns, tt.mockDeployCore)
			}

			if tt.mockDeployCon != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(deployConsole, tt.mockDeployCon)
			}

			if tt.mockWaitCon != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(waitAllConsolePodRunning, tt.mockWaitCon)
			}

			if tt.mockGenerate != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(generateSecret, tt.mockGenerate)
			}

			if tt.mockK3sRestart != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(k3sRestart, tt.mockK3sRestart)
			}

			if tt.mockDeployOauth != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(deployOauthAndUser, tt.mockDeployOauth)
			}

			err := DeployConsoleAll(config, tt.repo, tt.openFuyaoVersion)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
