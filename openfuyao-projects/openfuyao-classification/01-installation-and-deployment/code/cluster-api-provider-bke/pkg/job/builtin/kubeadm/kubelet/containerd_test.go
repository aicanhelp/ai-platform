/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package kubelet

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/containerd/containerd"
	"github.com/stretchr/testify/assert"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	containerd_executor "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

// 时间常量
const (
	shortWaitDuration    = 100 * time.Millisecond
	longWaitDuration     = 5 * time.Second
	defaultSleepDuration = 3 * time.Second
)

func TestRunWithContainerd(t *testing.T) {
	// 由于runWithContainerd依赖于executor和各种系统命令，我们使用gomonkey进行mock
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, _ map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return "/tmp/kubelet.sh"
	})

	patches.ApplyFunc(newKubeletScript, func(_ map[string]string) error {
		return nil
	})

	// 创建一个mock executor
	mockExec := &MockExecutor{}

	config := map[string]string{
		"containerName":     "kubelet",
		"dataRootDir":       "/var/lib/kubelet",
		"kubeletImage":      "k8s.gcr.io/kubelet:v1.24.0",
		"pauseImage":        "k8s.gcr.io/pause:3.9",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "/etc:/etc;/var:/var",
		"extraArgs":         "--max-pods=110",
		"kubernetesVersion": "v1.24.0",
		"hostIP":            "192.168.1.100",
		"imageRepo":         "k8s.gcr.io",
	}

	plugin := &kubeletPlugin{}
	plugin.exec = mockExec

	err := plugin.runWithContainerd(config)

	// 由于我们mock了所有依赖，应该成功执行
	assert.NoError(t, err)
}

// MockExecutor 是一个简单的executor mock
type MockExecutor struct{}

func (m *MockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *MockExecutor) ExecuteCommand(command string, args ...string) error {
	return nil
}

func (m *MockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *MockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *MockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *MockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *MockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *MockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *MockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *MockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestRemoveKubeletContainers(t *testing.T) {
	// Mock containerd client
	mockContainerd := &MockContainerdClient{}

	config := map[string]string{
		"imageRepo":         "k8s.gcr.io",
		"kubernetesVersion": "v1.24.0",
	}

	plugin := &kubeletPlugin{}
	plugin.containerd = mockContainerd

	err := plugin.removeKubeletContainers(config)

	// 由于我们mock了所有依赖，应该成功执行
	assert.NoError(t, err)
}

// MockContainerdClient 是一个简单的containerd client mock
type MockContainerdClient struct{}

func (m *MockContainerdClient) Pull(imageRef containerd_executor.ImageRef) error {
	return nil
}

func (m *MockContainerdClient) Close() {}

func (m *MockContainerdClient) Stop(containerId string) error {
	return nil
}

func (m *MockContainerdClient) Delete(containerId string) error {
	return nil
}

func (m *MockContainerdClient) Run(cs containerd_executor.ContainerSpec) error {
	return nil
}

func (m *MockContainerdClient) EnsureImageExists(image containerd_executor.ImageRef) error {
	return nil
}

func (m *MockContainerdClient) EnsureContainerRun(containerId string) (bool, error) {
	return false, nil
}

func (m *MockContainerdClient) ContainerList(filters ...string) ([]containerd.Container, error) {
	return []containerd.Container{}, nil
}

func (m *MockContainerdClient) Ping() error {
	return nil
}

func TestNerdctlRunKubeletCommand(t *testing.T) {
	// Mock外部依赖
	patches := gomonkey.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return "192.168.1.100", nil
	})
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	// Mock template content
	patches.ApplyGlobalVar(&nerdctl, "nerdctl -n {{.Namespace}} run -d --name {{.ContainerName}} {{.Image}}")

	config := map[string]string{
		"kubeletImage":      "k8s.gcr.io/kubelet:v1.24.0",
		"pauseImage":        "k8s.gcr.io/pause:3.9",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "/etc:/etc;/var:/var",
		"extraArgs":         "--max-pods=110",
		"kubernetesVersion": "v1.24.0",
		"hostIP":            "192.168.1.100",
	}

	cmd, args, err := nerdctlRunKubeletCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
}

func TestNewNerdctlCommand(t *testing.T) {
	config := map[string]string{
		"kubeletImage":      "k8s.gcr.io/kubelet:v1.24.0",
		"pauseImage":        "k8s.gcr.io/pause:3.9",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "/etc:/etc;/var:/var",
		"extraArgs":         "--max-pods=110",
		"kubernetesVersion": "v1.24.0",
		"hostIP":            "192.168.1.100",
		"dataRootDir":       "/var/lib/kubelet",
	}

	cmd, args, err := newNerdctlCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
}

func TestMountListForContainerd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	result := mountList()

	assert.NotEmpty(t, result)

	expectedMounts := []string{"/etc/kubernetes", "/var/lib/kubelet", "/etc/ssl/certs"}
	for _, expectedMount := range expectedMounts {
		found := false
		for _, mount := range result {
			if len(mount) > 1 && mount[0] == expectedMount {
				found = true
				break
			}
		}
		assert.True(t, found, "Expected mount path %s not found", expectedMount)
	}
}
