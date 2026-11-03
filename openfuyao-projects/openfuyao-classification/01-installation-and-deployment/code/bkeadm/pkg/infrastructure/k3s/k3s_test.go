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

package k3s

import (
	"context"
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	dockerapi "github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	fake "k8s.io/client-go/kubernetes/fake"
	k8sioTesting "k8s.io/client-go/testing"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testIPv4SegmentA = 127
	testIPv4SegmentB = 0
	testIPv4SegmentC = 0
	testIPv4SegmentD = 1

	testIPv4SegmentE = 192
	testIPv4SegmentF = 168
	testIPv4SegmentG = 1
	testIPv4SegmentH = 100

	testIPv4SegmentI = 172
	testIPv4SegmentJ = 17
	testIPv4SegmentK = 0
	testIPv4SegmentL = 2

	testIPv4SegmentM = 8
	testIPv4SegmentN = 8
	testIPv4SegmentO = 8
	testIPv4SegmentP = 8

	testPortValue = 6443
	testDirPath   = "/tmp/test_dir"
	testFilePath  = "/tmp/test_file"
	testFourValue = 4
	testSixValue  = 6
)

var testLoopbackIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
).String()

var testHostIP = net.IPv4(
	testIPv4SegmentE,
	testIPv4SegmentF,
	testIPv4SegmentG,
	testIPv4SegmentH,
).String()

var testContainerIP = net.IPv4(
	testIPv4SegmentI,
	testIPv4SegmentJ,
	testIPv4SegmentK,
	testIPv4SegmentL,
).String()

var testDNSIP = net.IPv4(
	testIPv4SegmentM,
	testIPv4SegmentN,
	testIPv4SegmentO,
	testIPv4SegmentP,
).String()

var testPort = fmt.Sprintf("%d", testPortValue)

func TestEnsureDirExists(t *testing.T) {
	tests := []struct {
		name        string
		dir         string
		mockExists  func(string) bool
		mockMkdir   func(string, os.FileMode) error
		expectedErr bool
	}{
		{
			name: "directory exists",
			dir:  testDirPath,
			mockExists: func(path string) bool {
				return true
			},
			mockMkdir: func(path string, perm os.FileMode) error {
				return nil
			},
			expectedErr: false,
		},
		{
			name: "directory does not exist, create success",
			dir:  testDirPath,
			mockExists: func(path string) bool {
				return false
			},
			mockMkdir: func(path string, perm os.FileMode) error {
				return nil
			},
			expectedErr: false,
		},
		{
			name: "directory does not exist, create fails",
			dir:  testDirPath,
			mockExists: func(path string) bool {
				return false
			},
			mockMkdir: func(path string, perm os.FileMode) error {
				return fmt.Errorf("mkdir failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.expectedErr {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdir)
			} else {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdir)
			}

			err := EnsureDirExists(tt.dir)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestEnsureK3sLogDirs(t *testing.T) {
	created := []string{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})
	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		created = append(created, path)
		return nil
	})

	assert.NoError(t, EnsureK3sLogDirs())
	assert.ElementsMatch(t, []string{K3sPodLogDir, K3sContainerLogDir}, created)
}

func TestIsKubernetesAvailable(t *testing.T) {
	tests := []struct {
		name           string
		setupMock      func() (*k8s.MockK8sClient, *gomonkey.Patches)
		mockNewClient  func(string) (k8s.KubernetesClient, error)
		expectedResult bool
	}{
		{
			name: "kubernetes available",
			setupMock: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Status.Conditions = []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
				}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient, patches
			},
			mockNewClient: func(kubeConfig string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			expectedResult: true,
		},
		{
			name: "kubernetes not available - client creation fails",
			setupMock: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				return &k8s.MockK8sClient{}, nil
			},
			mockNewClient: func(kubeConfig string) (k8s.KubernetesClient, error) {
				return nil, fmt.Errorf("client creation failed")
			},
			expectedResult: false,
		},
		{
			name: "kubernetes not available - no ready nodes",
			setupMock: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Status.Conditions = []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
				}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient, patches
			},
			mockNewClient: func(kubeConfig string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			expectedResult: false,
		},
		{
			name: "kubernetes not available - no nodes",
			setupMock: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				fakeClient := fake.NewSimpleClientset()
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient, patches
			},
			mockNewClient: func(kubeConfig string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient, setupPatches := tt.setupMock()
			if setupPatches != nil {
				defer setupPatches.Reset()
			}

			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeConfig string) (k8s.KubernetesClient, error) {
				client, err := tt.mockNewClient(kubeConfig)
				if err != nil {
					return nil, err
				}
				if setupPatches != nil && mockClient != nil {
					return mockClient, nil
				}
				return client, nil
			})

			originalK8s := global.K8s
			defer func() { global.K8s = originalK8s }()

			result := isKubernetesAvailable()

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestStartK3sWithContainerd(t *testing.T) {

	tests := []struct {
		name                string
		cfg                 Config
		localImage          string
		mockIsAvailable     func() bool
		mockPrepareImages   func(string, string, string, string, string)
		mockEnsureImage     func(string) error
		mockContainerRemove func(string) error
		mockExists          func(string) bool
		mockMkdir           func(string, os.FileMode) error
		mockCustomCA        func() error
		mockGetImageRepoIP  func(string, string, string, string) (string, string)
		mockGenerateConfig  func(string, string) error
		mockLogFormat       func(string, ...any)
		mockRun             func([]string) error
		mockSleep           func(time.Duration)
		mockCP              func(string, string) error
		mockSetupKubeconfig func(string, string) error
		expectedErr         bool
	}{
		{
			name: "kubernetes already available",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockIsAvailable: func() bool { return true },
			expectedErr:     false,
		},
		{
			name: "start k3s success",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
				ConsolePort:    "31000",
			},
			localImage:          "",
			mockIsAvailable:     func() bool { return false },
			mockPrepareImages:   func(string, string, string, string, string) {},
			mockEnsureImage:     func(string) error { return nil },
			mockContainerRemove: func(string) error { return nil },
			mockExists:          func(string) bool { return true },
			mockMkdir:           func(string, os.FileMode) error { return nil },
			mockCustomCA:        func() error { return nil },
			mockGetImageRepoIP:  func(string, string, string, string) (string, string) { return "test.repo:443", testLoopbackIP },
			mockGenerateConfig:  func(string, string) error { return nil },
			mockLogFormat:       func(string, ...any) {},
			mockRun: func(args []string) error {
				assert.Contains(t, args, "31000:30010")
				assert.Contains(t, args, fmt.Sprintf("%s:%s", K3sPodLogDir, K3sPodLogDir))
				assert.Contains(t, args, fmt.Sprintf("%s:%s", K3sContainerLogDir, K3sContainerLogDir))
				for _, arg := range args {
					assert.False(t, strings.HasPrefix(arg, "--token="), "k3s must not use a hardcoded --token")
				}
				return nil
			},
			mockSleep:           func(time.Duration) {},
			mockCP:              func(string, string) error { return nil },
			mockSetupKubeconfig: func(string, string) error { return nil },
			expectedErr:         false,
		},
		{
			name: "ensure image fails",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:          "",
			mockIsAvailable:     func() bool { return false },
			mockPrepareImages:   func(string, string, string, string, string) {},
			mockEnsureImage:     func(string) error { return fmt.Errorf("ensure image failed") },
			mockContainerRemove: func(string) error { return nil },
			expectedErr:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(isKubernetesAvailable, tt.mockIsAvailable)

			if tt.mockPrepareImages != nil {
				patches.ApplyFunc(prepareK3sImages, tt.mockPrepareImages)
			}
			if tt.mockEnsureImage != nil {
				patches.ApplyFunc(containerd.EnsureImageExists, func(string) error {
					return nil
				})
			}
			if tt.mockContainerRemove != nil {
				patches.ApplyFunc(containerd.ContainerRemove, tt.mockContainerRemove)
			}
			if tt.mockExists != nil {
				patches.ApplyFunc(utils.Exists, tt.mockExists)
			}
			if tt.mockMkdir != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdir)
			}
			if tt.mockCustomCA != nil {
				patches.ApplyFunc(customCA, tt.mockCustomCA)
			}
			if tt.mockGetImageRepoIP != nil {
				patches.ApplyFunc(getImageRepoIP, tt.mockGetImageRepoIP)
			}
			if tt.mockGenerateConfig != nil {
				patches.ApplyFunc(generateRegistriesConfig, tt.mockGenerateConfig)
			}
			if tt.mockLogFormat != nil {
				patches.ApplyFunc(log.SteppedInfo, tt.mockLogFormat)
			}
			if tt.mockRun != nil {
				patches.ApplyFunc(containerd.Run, tt.mockRun)
			}
			if tt.mockSleep != nil {
				patches.ApplyFunc(time.Sleep, tt.mockSleep)
			}
			if tt.mockCP != nil {
				patches.ApplyFunc(containerd.CP, tt.mockCP)
			}
			if tt.mockSetupKubeconfig != nil {
				patches.ApplyFunc(setupKubeconfigAndWaitCluster, tt.mockSetupKubeconfig)
			}

			err := StartK3sWithContainerd(tt.cfg, tt.localImage)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomCA(t *testing.T) {

	tests := []struct {
		name          string
		mockExists    func(string) bool
		mockMkdir     func(string, os.FileMode) error
		mockWriteFile func(string, []byte, os.FileMode) error
		mockExecute   func(*exec.CommandExecutor, string, ...string) (string, error)
		expectedErr   bool
	}{
		{
			name: "custom CA success",
			mockExists: func(string) bool {
				return false
			},
			mockMkdir: func(string, os.FileMode) error {
				return nil
			},
			mockWriteFile: func(string, []byte, os.FileMode) error {
				return nil
			},
			mockExecute: func(*exec.CommandExecutor, string, ...string) (string, error) {
				return "", nil
			},
			expectedErr: false,
		},
		{
			name: "mkdir fails",
			mockExists: func(string) bool {
				return false
			},
			mockMkdir: func(string, os.FileMode) error {
				return fmt.Errorf("mkdir failed")
			},
			expectedErr: true,
		},
		{
			name: "write file fails",
			mockExists: func(string) bool {
				return false
			},
			mockMkdir: func(string, os.FileMode) error {
				return nil
			},
			mockWriteFile: func(string, []byte, os.FileMode) error {
				return fmt.Errorf("write file failed")
			},
			expectedErr: true,
		},
		{
			name: "execute fails",
			mockExists: func(string) bool {
				return false
			},
			mockMkdir: func(string, os.FileMode) error {
				return nil
			},
			mockWriteFile: func(string, []byte, os.FileMode) error {
				return nil
			},
			mockExecute: func(*exec.CommandExecutor, string, ...string) (string, error) {
				return "execution failed", fmt.Errorf("execution failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdir)
			patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)

			mockExecutor := &exec.CommandExecutor{}
			patches.ApplyMethod(mockExecutor, "ExecuteCommandWithCombinedOutput", tt.mockExecute)

			err := customCA()

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestFixCoreDnsLoop(t *testing.T) {

	tests := []struct {
		name             string
		otherRepo        string
		imageRepo        string
		mockCheckConfig  func() error
		mockVerifyConfig func() error
		mockVerifyRun    func() error
		mockModImage     func(string, string) error
		mockExecute      func(*exec.CommandExecutor, string, ...string) (string, error)
		expectedErr      bool
	}{
		{
			name:      "fix coreDNS loop success",
			otherRepo: "test.repo",
			imageRepo: "registry.test.com",
			mockCheckConfig: func() error {
				return nil
			},
			mockVerifyConfig: func() error {
				return nil
			},
			mockVerifyRun: func() error {
				return nil
			},
			mockModImage: func(string, string) error {
				return nil
			},
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				return "", nil
			},
			expectedErr: false,
		},
		{
			name:      "check config fails",
			otherRepo: "test.repo",
			imageRepo: "registry.test.com",
			mockCheckConfig: func() error {
				return fmt.Errorf("check config failed")
			},
			expectedErr: true,
		},
		{
			name:      "verify config fails",
			otherRepo: "test.repo",
			imageRepo: "registry.test.com",
			mockCheckConfig: func() error {
				return nil
			},
			mockVerifyConfig: func() error {
				return fmt.Errorf("verify config failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(checkCurrentCoreDNSConfig, tt.mockCheckConfig)

			if tt.mockVerifyConfig != nil {
				patches.ApplyFunc(verifyCoreDNSConfig, tt.mockVerifyConfig)
			}
			if tt.mockVerifyRun != nil {
				patches.ApplyFunc(verifyCoreDNSRunning, tt.mockVerifyRun)
			}
			if tt.mockModImage != nil {
				patches.ApplyFunc(ModK3sCorednsImage, tt.mockModImage)
			}
			if tt.mockExecute != nil {
				mockExecutor := &exec.CommandExecutor{}
				patches.ApplyMethod(mockExecutor, "ExecuteCommandWithCombinedOutput", tt.mockExecute)
			}

			err := FixCoreDnsLoop(tt.otherRepo, tt.imageRepo)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestModCorednsConfigWithRetry(t *testing.T) {

	tests := []struct {
		name        string
		otherRepo   string
		imageRepo   string
		mockFixLoop func(string, string) error
		expectedErr bool
	}{
		{
			name:      "mod config success on first try",
			otherRepo: "test.repo",
			imageRepo: "registry.test.com",
			mockFixLoop: func(string, string) error {
				return nil
			},
			expectedErr: false,
		},
		{
			name:      "mod config fails after retries",
			otherRepo: "test.repo",
			imageRepo: "registry.test.com",
			mockFixLoop: func(string, string) error {
				return fmt.Errorf("fix loop failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(FixCoreDnsLoop, tt.mockFixLoop)

			patches.ApplyFunc(time.Sleep, func(time.Duration) {})
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := ModCorednsConfigWithRetry(tt.otherRepo, tt.imageRepo)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestModK3sCorednsImage(t *testing.T) {

	tests := []struct {
		name          string
		otherRepo     string
		imageRepo     string
		mockSleep     func(time.Duration)
		mockExecute   func(*exec.CommandExecutor, string, ...string) (string, error)
		mockLogFormat func(string, ...any)
		expectedErr   bool
	}{
		{
			name:      "mod coredns image success",
			otherRepo: "",
			imageRepo: "registry.test.com",
			mockSleep: func(time.Duration) {},
			mockExecute: func(*exec.CommandExecutor, string, ...string) (string, error) {
				return "True", nil
			},
			mockLogFormat: func(string, ...any) {},
			expectedErr:   false,
		},
		{
			name:      "execute command fails",
			otherRepo: "registry.test.com",
			imageRepo: "registry.test.com",
			mockSleep: func(time.Duration) {},
			mockExecute: func(*exec.CommandExecutor, string, ...string) (string, error) {
				return "True", fmt.Errorf("execution failed")
			},

			mockLogFormat: func(string, ...any) {},
			expectedErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(time.Sleep, tt.mockSleep)

			patches.ApplyFunc(log.SteppedInfo, tt.mockLogFormat)
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithCombinedOutput, tt.mockExecute)

			err := ModK3sCorednsImage(tt.otherRepo, tt.imageRepo)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartK3sWithDocker(t *testing.T) {
	tests := []struct {
		name                  string
		cfg                   Config
		localImage            string
		mockIsAvailable       func() bool
		mockPrepareEnv        func(Config, string) (string, string, error)
		mockLogFormat         func(string, ...any)
		mockBuildConfig       func(string) *container.Config
		mockBuildHostConfig   func(string, string, string) *container.HostConfig
		mockDockerRun         func(c *docker.Client, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, containerName string) error
		mockSleep             func(time.Duration)
		mockCopyFromContainer func(c *docker.Client, containerId, srcPath, dstPath string) error
		mockSetupKubeconfig   func(string, string) error
		expectedErr           bool
	}{
		{
			name: "kubernetes already available",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockIsAvailable: func() bool { return true },
			expectedErr:     false,
		},
		{
			name: "prepare environment fails",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockIsAvailable: func() bool { return false },
			mockPrepareEnv: func(Config, string) (string, string, error) {
				return "", "", fmt.Errorf("prepare env failed")
			},
			expectedErr: true,
		},
		{
			name: "start k3s with docker success",
			cfg: Config{
				OtherRepo:      "test.repo",
				OtherRepoIP:    testLoopbackIP,
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockIsAvailable: func() bool { return false },
			mockPrepareEnv: func(Config, string) (string, string, error) {
				return "test.repo:443", testLoopbackIP, nil
			},
			mockLogFormat: func(string, ...any) {},
			mockBuildConfig: func(string) *container.Config {
				return &container.Config{}
			},
			mockBuildHostConfig: func(string, string, string) *container.HostConfig {
				return &container.HostConfig{}
			},
			mockDockerRun: func(c *docker.Client, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, containerName string) error {
				return nil
			},
			mockSleep: func(time.Duration) {},
			mockCopyFromContainer: func(c *docker.Client, containerId, srcPath, dstPath string) error {
				return nil
			},
			mockSetupKubeconfig: func(string, string) error {
				return nil
			},
			expectedErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(isKubernetesAvailable, tt.mockIsAvailable)

			if tt.mockPrepareEnv != nil {
				patches.ApplyFunc(prepareK3sEnvironment, tt.mockPrepareEnv)
			}

			if tt.mockLogFormat != nil {
				patches.ApplyFunc(log.SteppedInfo, tt.mockLogFormat)
			}

			if tt.mockBuildConfig != nil {
				patches.ApplyFunc(buildK3sContainerConfig, tt.mockBuildConfig)
			}

			if tt.mockBuildHostConfig != nil {
				patches.ApplyFunc(buildK3sHostConfig, tt.mockBuildHostConfig)
			}

			if tt.mockDockerRun != nil {
				patches.ApplyFunc((*docker.Client).Run, tt.mockDockerRun)
			}

			if tt.mockSleep != nil {
				patches.ApplyFunc(time.Sleep, tt.mockSleep)
			}

			if tt.mockCopyFromContainer != nil {
				patches.ApplyFunc((*docker.Client).CopyFromContainer, tt.mockCopyFromContainer)
			}

			if tt.mockSetupKubeconfig != nil {
				patches.ApplyFunc(setupKubeconfigAndWaitCluster, tt.mockSetupKubeconfig)
			}

			if global.Docker == nil {
				global.Docker = &docker.Client{}
			}
			err := StartK3sWithDocker(tt.cfg, tt.localImage)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestReadKubeconfig(t *testing.T) {

	tests := []struct {
		name           string
		mockReadFile   func(string) ([]byte, error)
		expectedErr    bool
		expectedResult bool
	}{
		{
			name: "read kubeconfig success",
			mockReadFile: func(path string) ([]byte, error) {
				return []byte("test-kubeconfig-content"), nil
			},
			expectedErr:    false,
			expectedResult: true,
		},
		{
			name: "read kubeconfig fails",
			mockReadFile: func(path string) ([]byte, error) {
				return nil, fmt.Errorf("file not found")
			},
			expectedErr:    true,
			expectedResult: false,
		},
		{
			name: "read kubeconfig returns empty",
			mockReadFile: func(path string) ([]byte, error) {
				return []byte{}, nil
			},
			expectedErr:    true,
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.ReadFile, tt.mockReadFile)

			patches.ApplyFunc(time.Sleep, func(time.Duration) {})

			result, err := readKubeconfig()

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, result)
			}
		})
	}
}

func TestProcessKubeconfig(t *testing.T) {
	tests := []struct {
		name           string
		hostIP         string
		kubernetesPort string
		kubeconfigData []byte
		mockUserHome   func() (string, error)
		mockMkdirAll   func(string, os.FileMode) error
		mockWriteFile  func(string, []byte, os.FileMode) error
		mockRemove     func(string) error
		expectedErr    bool
	}{
		{
			name:           "process kubeconfig success",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			kubeconfigData: []byte("apiVersion: v1\nclusters:\n- cluster:\n    server: https://127.0.0.1:36443\n  name: default\ncontexts:\n- context:\n    cluster: default\n    user: default\n  name: default\ncurrent-context: default\nkind: Config\npreferences: {}\nusers:\n- name: default\n  user:\n    token: test-token"),
			mockUserHome:   func() (string, error) { return "/home/testuser", nil },
			mockMkdirAll:   func(string, os.FileMode) error { return nil },
			mockWriteFile:  func(string, []byte, os.FileMode) error { return nil },
			mockRemove:     func(string) error { return nil },
			expectedErr:    false,
		},
		{
			name:           "get user home fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			kubeconfigData: []byte("test-content"),
			mockUserHome:   func() (string, error) { return "", fmt.Errorf("get home failed") },
			expectedErr:    true,
		},
		{
			name:           "mkdir fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			kubeconfigData: []byte("test-content"),
			mockUserHome:   func() (string, error) { return "/home/testuser", nil },
			mockMkdirAll:   func(string, os.FileMode) error { return fmt.Errorf("mkdir failed") },
			expectedErr:    true,
		},
		{
			name:           "write kubeconfig fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			kubeconfigData: []byte("test-content"),
			mockUserHome:   func() (string, error) { return "/home/testuser", nil },
			mockMkdirAll:   func(string, os.FileMode) error { return nil },
			mockWriteFile:  func(string, []byte, os.FileMode) error { return fmt.Errorf("write failed") },
			expectedErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.UserHomeDir, tt.mockUserHome)

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)
			patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)
			patches.ApplyFunc(os.Remove, tt.mockRemove)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			result, err := processKubeconfig(tt.hostIP, tt.kubernetesPort, tt.kubeconfigData)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Contains(t, result, tt.hostIP)
			}
		})
	}
}

func TestWaitForK8sClient(t *testing.T) {

	tests := []struct {
		name           string
		kubeconfigPath string
		mockNewClient  func(string) (k8s.KubernetesClient, error)
		expectedErr    bool
	}{
		{
			name:           "wait for k8s client success",
			kubeconfigPath: "/home/testuser/.kube/config",
			mockNewClient: func(path string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			},
			expectedErr: false,
		},
		{
			name:           "wait for k8s client fails",
			kubeconfigPath: "/home/testuser/.kube/config",
			mockNewClient: func(path string) (k8s.KubernetesClient, error) {
				return nil, fmt.Errorf("client creation failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(k8s.NewKubernetesClient, tt.mockNewClient)

			patches.ApplyFunc(time.Sleep, func(time.Duration) {})

			originalK8s := global.K8s
			global.K8s = nil
			defer func() { global.K8s = originalK8s }()

			err := waitForK8sClient(tt.kubeconfigPath)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWaitForNodeReady(t *testing.T) {

	tests := []struct {
		name        string
		mockGetNode func(*k8s.MockK8sClient, string) error
		expectedErr bool
	}{
		{
			name: "node ready success",
			mockGetNode: func(client *k8s.MockK8sClient, name string) error {
				return nil
			},
			expectedErr: false,
		},
		{
			name: "node not ready",
			mockGetNode: func(client *k8s.MockK8sClient, name string) error {
				return fmt.Errorf("node not found")
			},
			expectedErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient
			defer func() { global.K8s = originalK8s }()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(time.Sleep, func(time.Duration) {})

			err := waitForNodeReady()

			assert.NoError(t, err)
			patches.Reset()
		})
	}
}

func TestCreateKubeconfigSecret(t *testing.T) {
	tests := []struct {
		name           string
		kubeconfigData string
		expectError    bool
	}{
		{
			name:           "create secret success",
			kubeconfigData: "test-kubeconfig-content",
			expectError:    false,
		},
		{
			name:           "create secret fails",
			kubeconfigData: "test-kubeconfig-content",
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() { global.K8s = originalK8s }()

			fakeClient := fake.NewSimpleClientset()

			if tt.expectError {
				fakeClient.PrependReactor("create", "secrets", func(action k8sioTesting.Action) (handled bool, ret k8sruntime.Object, err error) {
					return true, nil, fmt.Errorf("secret creation failed")
				})
			}

			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*k8s.MockK8sClient).GetClient, func(m *k8s.MockK8sClient) kubernetes.Interface {
				return fakeClient
			})

			err := createKubeconfigSecret(tt.kubeconfigData)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSetupKubeconfigAndWaitCluster(t *testing.T) {

	tests := []struct {
		name               string
		hostIP             string
		kubernetesPort     string
		mockReadKubeconfig func() ([]byte, error)
		mockProcess        func(string, string, []byte) (string, error)
		mockUserHome       func() (string, error)
		mockWaitClient     func(string) error
		mockWaitNode       func() error
		mockCreateSecret   func(string) error
		expectedErr        bool
	}{
		{
			name:           "setup kubeconfig success",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			mockReadKubeconfig: func() ([]byte, error) {
				return []byte("test-kubeconfig"), nil
			},
			mockProcess: func(hostIP, kubernetesPort string, data []byte) (string, error) {
				return "processed-kubeconfig", nil
			},
			mockUserHome:     func() (string, error) { return "/home/testuser", nil },
			mockWaitClient:   func(string) error { return nil },
			mockWaitNode:     func() error { return nil },
			mockCreateSecret: func(string) error { return nil },
			expectedErr:      false,
		},
		{
			name:           "read kubeconfig fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			mockReadKubeconfig: func() ([]byte, error) {
				return nil, fmt.Errorf("read failed")
			},
			expectedErr: true,
		},
		{
			name:           "process kubeconfig fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			mockReadKubeconfig: func() ([]byte, error) {
				return []byte("test-kubeconfig"), nil
			},
			mockProcess: func(hostIP, kubernetesPort string, data []byte) (string, error) {
				return "", fmt.Errorf("process failed")
			},
			expectedErr: true,
		},
		{
			name:           "wait client fails",
			hostIP:         testLoopbackIP,
			kubernetesPort: testPort,
			mockReadKubeconfig: func() ([]byte, error) {
				return []byte("test-kubeconfig"), nil
			},
			mockProcess: func(hostIP, kubernetesPort string, data []byte) (string, error) {
				return "processed-kubeconfig", nil
			},
			mockUserHome:   func() (string, error) { return "/home/testuser", nil },
			mockWaitClient: func(string) error { return fmt.Errorf("wait client failed") },
			expectedErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(readKubeconfig, tt.mockReadKubeconfig)

			if tt.mockProcess != nil {
				patches.ApplyFunc(processKubeconfig, tt.mockProcess)
			}
			if tt.mockUserHome != nil {
				patches.ApplyFunc(os.UserHomeDir, tt.mockUserHome)
			}
			if tt.mockWaitClient != nil {
				patches.ApplyFunc(waitForK8sClient, tt.mockWaitClient)
			}
			if tt.mockWaitNode != nil {
				patches.ApplyFunc(waitForNodeReady, tt.mockWaitNode)
			}
			if tt.mockCreateSecret != nil {
				patches.ApplyFunc(createKubeconfigSecret, tt.mockCreateSecret)
			}
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := setupKubeconfigAndWaitCluster(tt.hostIP, tt.kubernetesPort)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrepareK3sImages(t *testing.T) {

	tests := []struct {
		name          string
		otherRepo     string
		imageRepo     string
		imageRepoPort string
		localImage    string
	}{
		{
			name:          "with other repo",
			otherRepo:     "test.repo",
			imageRepo:     "registry.test.com",
			imageRepoPort: "5000",
			localImage:    "",
		},
		{
			name:          "without other repo",
			otherRepo:     "",
			imageRepo:     "registry.test.com",
			imageRepoPort: "5000",
			localImage:    "",
		},
		{
			name:          "with local image",
			otherRepo:     "",
			imageRepo:     "registry.test.com",
			imageRepoPort: "5000",
			localImage:    "local-image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prepareK3sImages("", tt.otherRepo, tt.imageRepoPort, tt.imageRepo, tt.localImage)

			if tt.otherRepo != "" && tt.localImage == "" {
				assert.Contains(t, k3sImage, tt.otherRepo)
			} else {
				assert.Contains(t, k3sImage, tt.imageRepoPort)
			}
		})
	}
}

func TestGetImageRepoIP(t *testing.T) {

	tests := []struct {
		name           string
		otherRepo      string
		otherRepoIP    string
		hostIP         string
		imageRepo      string
		localImagePath string
		mockInspect    func(string) (containerd.NerdContainerInfo, error)
		expectedRepo   string
		expectedIP     string
	}{
		{
			name:           "with other repo containing image repo",
			otherRepo:      "registry.test.com:5000/test",
			otherRepoIP:    testHostIP,
			hostIP:         testLoopbackIP,
			imageRepo:      "registry.test.com",
			localImagePath: "",
			mockInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{}, fmt.Errorf("container not found")
			},
			expectedRepo: "registry.test.com:5000",
			expectedIP:   testHostIP,
		},
		{
			name:           "with other repo not containing image repo",
			otherRepo:      "other.repo:5000/test",
			otherRepoIP:    testHostIP,
			hostIP:         testLoopbackIP,
			imageRepo:      "registry.test.com",
			localImagePath: "",
			mockInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{}, fmt.Errorf("container not found")
			},
			expectedRepo: "registry.test.com:443",
			expectedIP:   testLoopbackIP,
		},
		{
			name:           "inspect succeeds",
			otherRepo:      "",
			otherRepoIP:    "",
			hostIP:         testLoopbackIP,
			imageRepo:      "registry.test.com",
			localImagePath: "",
			mockInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{Id: "test-container", NetworkSettings: struct {
					IPAddress  string `json:"IPAddress"`
					MacAddress string `json:"MacAddress"`
				}{IPAddress: testContainerIP}}, nil
			},
			expectedRepo: "registry.test.com:443",
			expectedIP:   testContainerIP,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(containerd.ContainerInspect, tt.mockInspect)

			repo, ip := getImageRepoIP(tt.otherRepo, tt.otherRepoIP, tt.hostIP, tt.imageRepo, tt.localImagePath)

			assert.Equal(t, tt.expectedRepo, repo)
			assert.Equal(t, tt.expectedIP, ip)
		})
	}
}

func TestGetImageRepoIPWithDocker(t *testing.T) {

	const defaultImageRepo = "deploy.bocloud.k8s"

	tests := []struct {
		name                 string
		otherRepo            string
		otherRepoIP          string
		hostIP               string
		imageRepo            string
		mockContainerInspect func() error
		mockGetIP            string
		expectedRepo         string
		expectedIP           string
	}{
		{
			name:        "with other repo containing default image repo",
			otherRepo:   "deploy.bocloud.k8s:5000/test",
			otherRepoIP: testHostIP,
			hostIP:      testLoopbackIP,
			imageRepo:   "deploy.bocloud.k8s",
			mockContainerInspect: func() error {
				return nil
			},
			mockGetIP:    "",
			expectedRepo: "deploy.bocloud.k8s:5000",
			expectedIP:   testHostIP,
		},
		{
			name:        "inspect succeeds with IP",
			otherRepo:   "",
			otherRepoIP: "",
			hostIP:      testLoopbackIP,
			imageRepo:   "registry.test.com",
			mockContainerInspect: func() error {
				return nil
			},
			mockGetIP:    testContainerIP,
			expectedRepo: "registry.test.com:443",
			expectedIP:   testContainerIP,
		},
		{
			name:        "inspect fails with error",
			otherRepo:   "",
			otherRepoIP: "",
			hostIP:      testLoopbackIP,
			imageRepo:   "registry.test.com",
			mockContainerInspect: func() error {
				return fmt.Errorf("inspect failed")
			},
			mockGetIP:    "",
			expectedRepo: "registry.test.com:443",
			expectedIP:   testLoopbackIP,
		},
		{
			name:        "with other repo not containing default image repo",
			otherRepo:   "other.repo:5000/test",
			otherRepoIP: testHostIP,
			hostIP:      testLoopbackIP,
			imageRepo:   "registry.test.com",
			mockContainerInspect: func() error {
				return fmt.Errorf("container not found")
			},
			mockGetIP:    "",
			expectedRepo: "registry.test.com:443",
			expectedIP:   testLoopbackIP,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDockerClient := &docker.Client{}
			mockDockerPatches := gomonkey.ApplyMethod(mockDockerClient, "GetClient", func(_ *docker.Client) *dockerapi.Client {
				return nil
			})
			defer mockDockerPatches.Reset()

			containerInspectPatches := gomonkey.ApplyFunc((*dockerapi.Client).ContainerInspect,
				func(_ *dockerapi.Client, ctx context.Context, containerID string) (container.InspectResponse, error) {
					if tt.mockContainerInspect() != nil {
						return container.InspectResponse{}, tt.mockContainerInspect()
					}
					return container.InspectResponse{
						ContainerJSONBase: &container.ContainerJSONBase{
							ID: "test-container",
						},
						NetworkSettings: &container.NetworkSettings{
							DefaultNetworkSettings: container.DefaultNetworkSettings{
								IPAddress: tt.mockGetIP,
							},
						},
					}, nil
				})
			defer containerInspectPatches.Reset()

			originalDocker := global.Docker
			global.Docker = mockDockerClient
			defer func() { global.Docker = originalDocker }()

			_ = defaultImageRepo

			repo, ip := getImageRepoIPWithDocker(tt.otherRepo, tt.otherRepoIP, tt.hostIP, tt.imageRepo)

			assert.Equal(t, tt.expectedRepo, repo)
			assert.Equal(t, tt.expectedIP, ip)
		})
	}
}

func TestGenerateRegistriesConfig(t *testing.T) {

	tests := []struct {
		name          string
		mockWriteFile func(string, []byte, os.FileMode) error
		expectedErr   bool
	}{
		{
			name: "generate config success",
			mockWriteFile: func(string, []byte, os.FileMode) error {
				return nil
			},
			expectedErr: false,
		},
		{
			name: "generate config fails",
			mockWriteFile: func(string, []byte, os.FileMode) error {
				return fmt.Errorf("generate config failed")
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k3sConfig := t.TempDir()
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)

			err := generateRegistriesConfig("test.repo:443", k3sConfig)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCheckCurrentCoreDNSConfig(t *testing.T) {

	tests := []struct {
		name        string
		mockExecute func(*exec.CommandExecutor, string, ...string) (string, error)
		expectedErr bool
	}{
		{
			name: "config contains /etc/resolv.conf",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "/etc/resolv.conf nameserver " + testDNSIP, nil
				}
				return "", nil
			},
			expectedErr: false,
		},
		{
			name: "config contains fixed DNS",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "forward . " + testDNSIP, nil
				}
				return "", nil
			},
			expectedErr: false,
		},
		{
			name: "config is empty",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "", nil
				}
				return "", nil
			},
			expectedErr: false,
		},
		{
			name: "execute command fails",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "", fmt.Errorf("command failed")
				}
				return "", nil
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockExecutor := &exec.CommandExecutor{}
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyMethod(mockExecutor, "ExecuteCommandWithCombinedOutput", tt.mockExecute)

			err := checkCurrentCoreDNSConfig()

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestVerifyCoreDNSConfig(t *testing.T) {

	tests := []struct {
		name        string
		mockExecute func(*exec.CommandExecutor, string, ...string) (string, error)
		mockSleep   func(time.Duration)
		expectedErr bool
	}{
		{
			name: "verify config success",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "forward . " + testDNSIP, nil
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: false,
		},
		{
			name: "config still contains /etc/resolv.conf",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "/etc/resolv.conf nameserver " + testDNSIP, nil
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: true,
		},
		{
			name: "fixed DNS not found",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "some random config", nil
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: true,
		},
		{
			name: "execute command fails",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "", fmt.Errorf("command failed")
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockExecutor := &exec.CommandExecutor{}
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyMethod(mockExecutor, "ExecuteCommandWithCombinedOutput", tt.mockExecute)

			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := verifyCoreDNSConfig()

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestVerifyCoreDNSRunning(t *testing.T) {

	tests := []struct {
		name        string
		mockExecute func(*exec.CommandExecutor, string, ...string) (string, error)
		mockSleep   func(time.Duration)
		expectedErr bool
	}{
		{
			name: "coredns running",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "Running", nil
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: false,
		},
		{
			name: "coredns not running",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "Pending", nil
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: true,
		},
		{
			name: "execute command fails",
			mockExecute: func(executor *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				if len(args) > 0 && args[0] == "get" {
					return "", fmt.Errorf("command failed")
				}
				return "", nil
			},
			mockSleep:   func(time.Duration) {},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockExecutor := &exec.CommandExecutor{}
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyMethod(mockExecutor, "ExecuteCommandWithCombinedOutput", tt.mockExecute)

			patches.ApplyFunc(time.Sleep, tt.mockSleep)
			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			err := verifyCoreDNSRunning()

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestContainsFixedDNS(t *testing.T) {

	tests := []struct {
		name     string
		config   string
		expected bool
	}{
		{
			name:     "contains forward . 8.8.8.8",
			config:   "forward . " + testDNSIP,
			expected: true,
		},
		{
			name:     "contains forward . 8.8.4.4",
			config:   "forward . 8.8.4.4",
			expected: true,
		},
		{
			name:     "contains forward . 1.1.1.1",
			config:   "forward . 1.1.1.1",
			expected: true,
		},
		{
			name:     "contains forward . 1.0.0.1",
			config:   "forward . 1.0.0.1",
			expected: true,
		},
		{
			name:     "contains forward . 208.67.222.222",
			config:   "forward . 208.67.222.222",
			expected: true,
		},
		{
			name:     "contains forward . 208.67.220.220",
			config:   "forward . 208.67.220.220",
			expected: true,
		},
		{
			name:     "no fixed DNS",
			config:   "some random config",
			expected: false,
		},
		{
			name:     "empty config",
			config:   "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containsFixedDNS(tt.config)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildK3sContainerConfig(t *testing.T) {

	tests := []struct {
		name   string
		hostIP string
	}{
		{
			name:   "build container config",
			hostIP: testLoopbackIP,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := buildK3sContainerConfig(tt.hostIP)

			assert.NotNil(t, config)
			assert.Equal(t, utils.LocalKubernetesName, config.Hostname)
			assert.Contains(t, config.Cmd, "--snapshotter=native")
			assert.Contains(t, config.Cmd, "--service-cidr=100.10.0.0/16")
			assert.Contains(t, config.Cmd, "--cluster-cidr=100.20.0.0/16")
			assert.Contains(t, config.Cmd, fmt.Sprintf("--tls-san=%s", tt.hostIP))
			assert.Contains(t, config.Cmd, fmt.Sprintf("--node-name=%s", utils.LocalKubernetesName))
			for _, arg := range config.Cmd {
				assert.False(t, strings.HasPrefix(arg, "--token="), "k3s must not use a hardcoded --token")
			}
			assert.NotNil(t, config.ExposedPorts)
			assert.Contains(t, config.ExposedPorts, nat.Port("6443/tcp"))
		})
	}
}

func TestBuildK3sHostConfig(t *testing.T) {

	tests := []struct {
		name           string
		kubernetesPort string
		imageRepo      string
		imageRepoIP    string
	}{
		{
			name:           "build host config",
			kubernetesPort: testPort,
			imageRepo:      "registry.test.com",
			imageRepoIP:    testLoopbackIP,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := buildK3sHostConfig(tt.kubernetesPort, tt.imageRepo, tt.imageRepoIP)

			assert.NotNil(t, config)
			assert.NotNil(t, config.PortBindings)
			assert.Contains(t, config.PortBindings, nat.Port("6443/tcp"))
			assert.Equal(t, tt.kubernetesPort, config.PortBindings[nat.Port("6443/tcp")][0].HostPort)
			assert.Equal(t, container.RestartPolicy{Name: "on-failure", MaximumRetryCount: 10}, config.RestartPolicy)
			assert.Contains(t, config.ExtraHosts, fmt.Sprintf("%s:%s", tt.imageRepo, tt.imageRepoIP))
			assert.True(t, config.Privileged)
			assert.Contains(t, config.SecurityOpt, "seccomp=unconfined")
			assert.Contains(t, config.SecurityOpt, "apparmor=unconfined")
			assert.Contains(t, config.SecurityOpt, "label=disable")
			assert.NotNil(t, config.Mounts)
			assert.Len(t, config.Mounts, testSixValue)
			assert.Contains(t, config.Mounts, mount.Mount{Type: mount.TypeBind, Source: K3sPodLogDir, Target: K3sPodLogDir})
			assert.Contains(t, config.Mounts, mount.Mount{Type: mount.TypeBind, Source: K3sContainerLogDir, Target: K3sContainerLogDir})
		})
	}
}

func TestPrepareK3sEnvironment(t *testing.T) {

	tests := []struct {
		name                string
		cfg                 Config
		localImage          string
		mockEnsureImage     func(image docker.ImageRef, options utils.RetryOptions) error
		mockEnsureRun       func(containerID string) (bool, error)
		mockContainerRemove func(containerID string) error
		mockExists          func(path string) bool
		mockMkdir           func(path string, perm os.FileMode) error
		mockCustomCA        func() error
		mockGetImageRepoIP  func(otherRepo, otherRepoIP, hostIP, imageRepo string) (string, string)
		mockGenerateConfig  func(repo, k3sConfig string) error
		mockIsAvailable     func() bool
		expectedRepo        string
		expectedIP          string
		expectError         bool
	}{
		{
			name: "kubernetes already available",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:          "",
			mockEnsureImage:     func(image docker.ImageRef, options utils.RetryOptions) error { return nil },
			mockEnsureRun:       func(containerID string) (bool, error) { return true, nil },
			mockContainerRemove: func(containerID string) error { return nil },
			mockExists:          func(path string) bool { return true },
			mockMkdir:           func(path string, perm os.FileMode) error { return nil },
			mockCustomCA:        func() error { return nil },
			mockGetImageRepoIP: func(otherRepo, otherRepoIP, hostIP, imageRepo string) (string, string) {
				return "test.repo:443", testLoopbackIP
			},
			mockGenerateConfig: func(repo, k3sConfig string) error { return nil },
			mockIsAvailable:    func() bool { return true },
			expectedRepo:       "",
			expectedIP:         "",
			expectError:        false,
		},
		{
			name: "prepare environment success",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:          "",
			mockEnsureImage:     func(image docker.ImageRef, options utils.RetryOptions) error { return nil },
			mockEnsureRun:       func(containerID string) (bool, error) { return false, nil },
			mockContainerRemove: func(containerID string) error { return nil },
			mockExists:          func(path string) bool { return true },
			mockMkdir:           func(path string, perm os.FileMode) error { return nil },
			mockCustomCA:        func() error { return nil },
			mockGetImageRepoIP: func(otherRepo, otherRepoIP, hostIP, imageRepo string) (string, string) {
				return "test.repo:443", testLoopbackIP
			},
			mockGenerateConfig: func(repo, k3sConfig string) error { return nil },
			mockIsAvailable:    func() bool { return false },
			expectedRepo:       "test.repo:443",
			expectedIP:         testLoopbackIP,
			expectError:        false,
		},
		{
			name: "ensure image fails",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage: "",
			mockEnsureImage: func(image docker.ImageRef, options utils.RetryOptions) error {
				return fmt.Errorf("ensure image failed")
			},
			expectError: true,
		},
		{
			name: "mkdir fails",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockEnsureImage: func(image docker.ImageRef, options utils.RetryOptions) error { return nil },
			mockEnsureRun:   func(containerID string) (bool, error) { return false, nil },
			mockExists:      func(path string) bool { return false },
			mockMkdir: func(path string, perm os.FileMode) error {
				return fmt.Errorf("mkdir failed")
			},
			expectError: true,
		},
		{
			name: "custom CA fails",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:      "",
			mockEnsureImage: func(image docker.ImageRef, options utils.RetryOptions) error { return nil },
			mockEnsureRun:   func(containerID string) (bool, error) { return false, nil },
			mockExists:      func(path string) bool { return true },
			mockMkdir:       func(path string, perm os.FileMode) error { return nil },
			mockCustomCA: func() error {
				return fmt.Errorf("custom CA failed")
			},
			expectError: true,
		},
		{
			name: "generate config fails",
			cfg: Config{
				OtherRepo:      "",
				OtherRepoIP:    "",
				HostIP:         testLoopbackIP,
				ImageRepo:      "registry.test.com",
				ImageRepoPort:  "5000",
				KubernetesPort: testPort,
			},
			localImage:          "",
			mockEnsureImage:     func(image docker.ImageRef, options utils.RetryOptions) error { return nil },
			mockEnsureRun:       func(containerID string) (bool, error) { return false, nil },
			mockContainerRemove: func(containerID string) error { return nil },
			mockExists:          func(path string) bool { return true },
			mockMkdir:           func(path string, perm os.FileMode) error { return nil },
			mockCustomCA:        func() error { return nil },
			mockGetImageRepoIP: func(otherRepo, otherRepoIP, hostIP, imageRepo string) (string, string) {
				return "test.repo:443", testLoopbackIP
			},
			mockGenerateConfig: func(repo, k3sConfig string) error {
				return fmt.Errorf("generate config failed")
			},
			mockIsAvailable: func() bool { return false },
			expectError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			if tt.mockMkdir != nil {
				patches.ApplyFunc(os.MkdirAll, tt.mockMkdir)
			}

			if tt.mockCustomCA != nil {
				patches.ApplyFunc(customCA, tt.mockCustomCA)
			}

			if tt.mockGetImageRepoIP != nil {
				patches.ApplyFunc(getImageRepoIPWithDocker, tt.mockGetImageRepoIP)
			}

			if tt.mockGenerateConfig != nil {
				patches.ApplyFunc(generateRegistriesConfig, tt.mockGenerateConfig)
			}

			if tt.mockIsAvailable != nil {
				patches.ApplyFunc(isKubernetesAvailable, tt.mockIsAvailable)
			}

			mockDockerClient := &docker.Client{}
			dockerPatches := gomonkey.ApplyMethod(mockDockerClient, "EnsureImageExists",
				func(_ *docker.Client, image docker.ImageRef, options utils.RetryOptions) error {
					return tt.mockEnsureImage(image, options)
				})
			defer dockerPatches.Reset()

			dockerPatches.ApplyMethod(mockDockerClient, "EnsureContainerRun",
				func(_ *docker.Client, containerID string) (bool, error) {
					return tt.mockEnsureRun(containerID)
				})

			if tt.mockContainerRemove != nil {
				dockerPatches.ApplyMethod(mockDockerClient, "ContainerRemove",
					func(_ *docker.Client, containerID string) error {
						return tt.mockContainerRemove(containerID)
					})
			}

			originalDocker := global.Docker
			global.Docker = mockDockerClient
			defer func() { global.Docker = originalDocker }()

			repo, ip, err := prepareK3sEnvironment(tt.cfg, tt.localImage)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.expectedRepo != "" {
					assert.Equal(t, tt.expectedRepo, repo)
					assert.Equal(t, tt.expectedIP, ip)
				}
			}
		})
	}
}

func TestWaitForNodeReadyWithK8sClient(t *testing.T) {

	tests := []struct {
		name           string
		setupK8sClient func() *k8s.MockK8sClient
		mockGetNode    func(*k8s.MockK8sClient, string) error
		expectedErr    bool
	}{
		{
			name: "node ready success",
			setupK8sClient: func() *k8s.MockK8sClient {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Spec.Taints = []corev1.Taint{}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				_, _ = fakeClient.CoreV1().Namespaces().Create(context.Background(), &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{Name: "kube-system"},
				}, metav1.CreateOptions{})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient
			},
			expectedErr: false,
		},
		{
			name: "node not ready - taints present",
			setupK8sClient: func() *k8s.MockK8sClient {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Spec.Taints = []corev1.Taint{{}}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient
			},
			expectedErr: false,
		},
		{
			name: "kube-system namespace not ready",
			setupK8sClient: func() *k8s.MockK8sClient {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Spec.Taints = []corev1.Taint{}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient
			},
			expectedErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() { global.K8s = originalK8s }()

			mockClient := tt.setupK8sClient()
			global.K8s = mockClient

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(time.Sleep, func(time.Duration) {})

			err := waitForNodeReady()

			assert.NoError(t, err)
		})
	}
}

func TestCreateKubeconfigSecretWithK8sClient(t *testing.T) {

	tests := []struct {
		name           string
		kubeconfigData string
		setupK8sClient func() (*k8s.MockK8sClient, *gomonkey.Patches)
		expectedErr    bool
	}{
		{
			name:           "create secret success",
			kubeconfigData: "test-kubeconfig-content",
			setupK8sClient: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				fakeClient := fake.NewSimpleClientset()
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient, patches
			},
			expectedErr: false,
		},
		{
			name:           "create secret fails",
			kubeconfigData: "test-kubeconfig-content",
			setupK8sClient: func() (*k8s.MockK8sClient, *gomonkey.Patches) {
				fakeClient := fake.NewSimpleClientset()
				fakeClient.PrependReactor("create", "secrets", func(action k8sioTesting.Action) (handled bool, ret k8sruntime.Object, err error) {
					return true, nil, fmt.Errorf("secret creation failed")
				})
				mockClient := &k8s.MockK8sClient{}
				patches := gomonkey.NewPatches()
				patches.ApplyMethod(mockClient, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
					return fakeClient
				})
				return mockClient, patches
			},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalK8s := global.K8s
			defer func() { global.K8s = originalK8s }()

			mockClient, patches := tt.setupK8sClient()
			defer patches.Reset()

			global.K8s = mockClient

			err := createKubeconfigSecret(tt.kubeconfigData)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestIsKubernetesAvailableWithRealK8sClient(t *testing.T) {

	tests := []struct {
		name           string
		setupFakeK8s   func() kubernetes.Interface
		expectedResult bool
	}{
		{
			name: "kubernetes available - node ready",
			setupFakeK8s: func() kubernetes.Interface {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Status.Conditions = []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
				}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				return fakeClient
			},
			expectedResult: true,
		},
		{
			name: "kubernetes not available - no ready nodes",
			setupFakeK8s: func() kubernetes.Interface {
				fakeClient := fake.NewSimpleClientset()
				node := &corev1.Node{}
				node.Name = utils.LocalKubernetesName
				node.Status.Conditions = []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
				}
				_, _ = fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
				return fakeClient
			},
			expectedResult: false,
		},
		{
			name: "kubernetes not available - no nodes",
			setupFakeK8s: func() kubernetes.Interface {
				return fake.NewSimpleClientset()
			},
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := tt.setupFakeK8s()

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyMethod(&k8s.MockK8sClient{}, "GetClient", func(_ *k8s.MockK8sClient) kubernetes.Interface {
				return fakeClient
			})

			patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeConfig string) (k8s.KubernetesClient, error) {
				return &k8s.MockK8sClient{}, nil
			})

			originalK8s := global.K8s
			defer func() { global.K8s = originalK8s }()

			result := isKubernetesAvailable()

			assert.Equal(t, tt.expectedResult, result)
		})
	}
}
