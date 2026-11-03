/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package kubelet

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types/container"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
)

const (
	testContainerName     = "test-kubelet"
	testDataRootDir       = "/var/lib/kubelet"
	testKubeletImage      = "registry.io/kubelet:v1.23.17"
	testPauseImage        = "registry.io/pause:3.9"
	testKubeconfigPath    = "/etc/kubernetes/admin.conf"
	testHostIP            = "192.168.1.100"
	testExtraVolumes      = "/data:/data;/config:/config"
	testExtraArgs         = "--v=3"
	testKubernetesVersion = "v1.23.17"
	testImageRepo         = "registry.io"
	testScriptPath        = "/etc/kubernetes/kubelet.sh"
	testPhase             = "bootstrap"
)

func TestNewDockerCommandBasic(t *testing.T) {
	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
	}

	cmd, args, err := newDockerCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
	assert.Contains(t, args, "--pod-infra-container-image="+testPauseImage)
	assert.Contains(t, args, "--kubeconfig="+testKubeconfigPath)
}

func TestNewDockerCommandWithExtraVolumes(t *testing.T) {
	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      testExtraVolumes,
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
	}

	cmd, args, err := newDockerCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
	allArgs := strings.Join(args, " ")
	assert.Contains(t, allArgs, "/data:/data")
	assert.Contains(t, allArgs, "/config:/config")
}

func TestNewDockerCommandWithExtraArgs(t *testing.T) {
	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         testExtraArgs,
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
	}

	cmd, args, err := newDockerCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
	assert.Contains(t, args, "--v=3")
}

func TestDockerRunKubeletCommandSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	externalIP := "10.0.0.100"
	testHostName := "test-node"

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return externalIP, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	config := map[string]string{
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"dataRootDir":       testDataRootDir,
	}

	cmd, args, err := dockerRunKubeletCommand(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, cmd)
	assert.NotEmpty(t, args)
	assert.Equal(t, "docker", cmd)
	assert.Contains(t, args, "run")
	assert.Contains(t, args, "-d")
}

func TestDockerRunKubeletCommandGetExternalIPError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return "", errors.New("failed to get external IP")
	})

	config := map[string]string{
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"dataRootDir":       testDataRootDir,
	}

	cmd, args, err := dockerRunKubeletCommand(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get external IP")
	assert.Empty(t, cmd)
	assert.Nil(t, args)
}

func TestGetDockerRunKubeletConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mountList, func() [][]string {
		return [][]string{
			{"/var/lib/kubelet", "/var/lib/kubelet"},
			{"/etc/kubernetes", "/etc/kubernetes"},
		}
	})

	config := map[string]string{
		"containerName":  testContainerName,
		"kubeletImage":   testKubeletImage,
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"dataRootDir":    testDataRootDir,
	}

	spec := getDockerRunKubeletConfig(config)

	assert.Equal(t, testContainerName, spec.ContainerName)
	assert.Equal(t, testKubeletImage, spec.ContainerConfig.Image)
	assert.True(t, spec.ContainerConfig.Tty)
	assert.True(t, spec.HostConfig.Privileged)
	assert.Equal(t, container.NetworkMode("host"), spec.HostConfig.NetworkMode)
	assert.Equal(t, container.PidMode("host"), spec.HostConfig.PidMode)
	assert.Equal(t, container.RestartPolicyMode("always"), spec.HostConfig.RestartPolicy.Name)
	assert.Len(t, spec.HostConfig.Mounts, 2)
}

func TestGetDockerRunKubeletConfigEmptyMounts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(mountList, func() [][]string {
		return [][]string{}
	})

	config := map[string]string{
		"containerName":  testContainerName,
		"kubeletImage":   testKubeletImage,
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"dataRootDir":    testDataRootDir,
	}

	spec := getDockerRunKubeletConfig(config)

	assert.Equal(t, testContainerName, spec.ContainerName)
	assert.Empty(t, spec.HostConfig.Mounts)
}

func TestCmdListBasic(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	externalIP := "192.168.1.50"
	testHostName := "test-node"

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return externalIP, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	config := map[string]string{
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"extraArgs":      "",
	}

	cmd := cmdList(config)

	assert.NotEmpty(t, cmd)
	assert.Equal(t, "kubelet", cmd[0])
	assert.Contains(t, cmd, "--pod-infra-container-image="+testPauseImage)
	assert.Contains(t, cmd, "--kubeconfig="+testKubeconfigPath)
	assert.Contains(t, cmd, "--node-ip="+externalIP)
	assert.Contains(t, cmd, "--hostname-override="+testHostName)
	assert.Contains(t, cmd, "--network-plugin=cni")
	assert.Contains(t, cmd, "--cni-conf-dir=/etc/cni/net.d")
	assert.Contains(t, cmd, "--cni-bin-dir=/opt/cni/bin")
}

func TestCmdListWithExtraArgs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	externalIP := "192.168.1.50"
	testHostName := "test-node"

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return externalIP, nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return testHostName
	})

	config := map[string]string{
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"extraArgs":      "--v=3 --test-flag",
	}

	cmd := cmdList(config)

	assert.NotEmpty(t, cmd)
	assert.Contains(t, cmd, "--v=3")
	assert.Contains(t, cmd, "--test-flag")
}

func TestCmdListGetExternalIPError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return "", errors.New("get IP error")
	})

	config := map[string]string{
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"extraArgs":      "",
	}

	cmd := cmdList(config)

	assert.NotEmpty(t, cmd)
	assert.Equal(t, "kubelet", cmd[0])
	assert.Contains(t, cmd, "--pod-infra-container-image="+testPauseImage)
}

func TestCmdListEmptyExtraArgs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return "10.0.0.1", nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "node1"
	})

	config := map[string]string{
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"extraArgs":      "",
	}

	cmd := cmdList(config)

	foundExtraArgs := false
	for _, arg := range cmd {
		if arg == "--v=3" || arg == "--test-flag" {
			foundExtraArgs = true
			break
		}
	}
	assert.False(t, foundExtraArgs)
}

func TestCmdListExtraArgsWithEmptyStrings(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(bkenet.GetExternalIP, func() (string, error) {
		return "10.0.0.1", nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "node1"
	})

	config := map[string]string{
		"pauseImage":     testPauseImage,
		"kubeconfigPath": testKubeconfigPath,
		"extraArgs":      "  --v=3   --test-flag  ",
	}

	cmd := cmdList(config)

	assert.NotEmpty(t, cmd)
	assert.Contains(t, cmd, "--v=3")
	assert.Contains(t, cmd, "--test-flag")
}

func TestEnsureDockerConfigSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	kp := &kubeletPlugin{
		exec: &mockExecutor{},
	}

	patches.ApplyFunc(docker.ConfigInsecureRegistries, func(registries []string) error {
		return nil
	})

	patches.ApplyFunc(docker.ConfigCgroupDriver, func(driver string) error {
		return nil
	})

	execCalls := 0
	patches.ApplyFunc(kp.exec.ExecuteCommandWithOutput, func(command string, arg ...string) (string, error) {
		execCalls++
		if execCalls == 1 {
			assert.Equal(t, "systemctl", command)
			assert.Contains(t, arg, "daemon-reload")
			return "", nil
		}
		assert.Equal(t, "systemctl", command)
		assert.Contains(t, arg, "restart")
		assert.Contains(t, arg, "docker")
		return "", nil
	})

	patches.ApplyFunc(docker.WaitDockerReady, func() error {
		return nil
	})

	config := map[string]string{
		"imageRepo": testImageRepo,
	}

	err := kp.ensureDockerConfig(config)

	assert.NoError(t, err)
}

func TestEnsureDockerConfigInsecureRegistriesError(t *testing.T) {
	kp := &kubeletPlugin{
		exec: &mockExecutor{},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(docker.ConfigInsecureRegistries, func(registries []string) error {
		return errors.New("config insecure registries failed")
	})

	config := map[string]string{
		"imageRepo": testImageRepo,
	}

	err := kp.ensureDockerConfig(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to ensure insecure registries")
}

func TestEnsureDockerConfigCgroupDriverError(t *testing.T) {
	kp := &kubeletPlugin{
		exec: &mockExecutor{},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(docker.ConfigInsecureRegistries, func(registries []string) error {
		return nil
	})

	patches.ApplyFunc(docker.ConfigCgroupDriver, func(driver string) error {
		return errors.New("config cgroup driver failed")
	})

	config := map[string]string{
		"imageRepo": testImageRepo,
	}

	err := kp.ensureDockerConfig(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to ensure cgroup driver")
}

type mockExecutor struct{}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

type trackingExecutor struct {
	mockExecutor
	execCalls []string
}

func (m *trackingExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *trackingExecutor) ExecuteCommand(command string, arg ...string) error {
	m.execCalls = append(m.execCalls, command+" "+strings.Join(arg, " "))
	return nil
}

func (m *trackingExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	m.execCalls = append(m.execCalls, command+" "+strings.Join(arg, " "))
	return "", nil
}

func (m *trackingExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

type errorExecutor struct {
	mockExecutor
}

func (m *errorExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "", errors.New("command execution failed")
}

func (m *errorExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func TestRunWithDockerSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "centos", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return nil
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kp.runWithDocker(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, exec.execCalls)
}

func TestRunWithDockerKylinWithDockerCE(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "kylin", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return nil
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kp.runWithDocker(config)

	assert.NoError(t, err)
	assert.NotEmpty(t, exec.execCalls)
}

func TestRunWithDockerKylinWithoutDockerCE(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "kylin", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return errors.New("repo search failed")
	})

	var generatedConfig map[string]string
	patches.ApplyFunc((*kubeletPlugin).generateKubeletConfig, func(_ *kubeletPlugin, config map[string]string) error {
		generatedConfig = config
		return nil
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kp.runWithDocker(config)

	assert.NoError(t, err)
	assert.NotNil(t, generatedConfig)
	assert.Equal(t, "cgroupfs", generatedConfig["cgroupDriver"])
}

func TestRunWithDockerGetPlatformInfoError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "", "", "", errors.New("failed to get platform info")
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kp.runWithDocker(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get host platform information failed")
}

func TestRunWithDockerEnsureImagesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "centos", "", "", nil
	})

	patches.ApplyFunc(kp.ensureImages, func(config map[string]string) error {
		return errors.New("failed to ensure images")
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
	}

	err := kp.runWithDocker(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to ensure images")
}

func TestRunWithDockerGenerateKubeletScriptError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	exec := &trackingExecutor{execCalls: []string{}}
	kp := &kubeletPlugin{
		exec: exec,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "centos", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return nil
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return errors.New("failed to generate kubelet script")
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kp.runWithDocker(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to generate kubelet script")
}

func TestRunWithDockerExecuteCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execErr := &errorExecutor{}
	kpErr := &kubeletPlugin{
		exec: execErr,
	}

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "centos", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return nil
	})

	patches.ApplyFunc(newKubeletScript, func(config map[string]string) error {
		return nil
	})

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testScriptPath
	})

	patches.ApplyFunc((*kubeletPlugin).ensureImages, func(_ *kubeletPlugin, config map[string]string) error {
		config["kubeletImage"] = testKubeletImage
		config["pauseImage"] = testPauseImage
		return nil
	})

	config := map[string]string{
		"containerName":     testContainerName,
		"kubeletImage":      testKubeletImage,
		"pauseImage":        testPauseImage,
		"kubeconfigPath":    testKubeconfigPath,
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": testKubernetesVersion,
		"hostIP":            testHostIP,
		"dataRootDir":       testDataRootDir,
		"imageRepo":         testImageRepo,
		"phase":             testPhase,
	}

	err := kpErr.runWithDocker(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to run kubelet container")
}
