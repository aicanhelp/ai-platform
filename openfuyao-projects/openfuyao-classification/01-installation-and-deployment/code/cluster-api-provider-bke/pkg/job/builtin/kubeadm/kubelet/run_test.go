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
	"context"
	"fmt"
	"os"
	"os/exec"
	executor "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	Runtime "k8s.io/apimachinery/pkg/runtime"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/download"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/runtime"
)

const (
	testProviderID        = "test-provider-id"
	testConfigPath        = "/var/lib/kubelet/config.yaml"
	testWorkspace         = "/workspace"
	testkubeletConfigMap  = "kube-system/kubelet-config"
	testkubeletConfigName = "test-kubelet-config"
	testkubeletConfigNS   = "bke-system"
	testkubeletPhase      = "bootstrap"
)

type testExecutor struct {
	execCalls   []string
	outputMap   map[string]string
	shouldFail  bool
	failCommand string
	output      string
}

func (e *testExecutor) WithContext(_ context.Context) executor.Executor {
	return e
}

func (e *testExecutor) ExecuteCommand(command string, arg ...string) error {
	e.execCalls = append(e.execCalls, command+" "+strings.Join(arg, " "))
	if e.shouldFail && strings.Contains(command, e.failCommand) {
		return errors.New("command failed")
	}
	return nil
}

func (e *testExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (e *testExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	e.execCalls = append(e.execCalls, command+" "+strings.Join(arg, " "))
	fullCmd := command + " " + strings.Join(arg, " ")
	if e.shouldFail && strings.Contains(fullCmd, e.failCommand) {
		return "", errors.New("command failed")
	}
	if val, ok := e.outputMap[command]; ok {
		return val, nil
	}
	return e.output, nil
}

func (e *testExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	e.execCalls = append(e.execCalls, command+" "+strings.Join(arg, " "))
	if e.shouldFail && strings.Contains(command, e.failCommand) {
		return "", errors.New("command failed")
	}
	return "", nil
}

func (e *testExecutor) ExecuteCommandWithCombinedOutputInDir(dir string, command string, arg ...string) (string, error) {
	e.execCalls = append(e.execCalls, fmt.Sprintf("dir=%q %s %s", dir, command, strings.Join(arg, " ")))
	if e.shouldFail && strings.Contains(command, e.failCommand) {
		return "", errors.New("command failed")
	}
	cmd := exec.Command(command, arg...)
	if dir != "" {
		cmd.Dir = dir
	}
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func (e *testExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestNewPlugin(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetDefaultImageRepo, func() string {
		return testImageRepo
	})

	kp := New(nil, nil)

	assert.NotNil(t, kp)
	assert.Equal(t, Name, kp.Name())
}

func TestName(t *testing.T) {
	kp := &kubeletPlugin{}
	assert.Equal(t, Name, kp.Name())
}

func TestParam(t *testing.T) {
	kp := &kubeletPlugin{}
	params := kp.Param()

	assert.NotEmpty(t, params)
	assert.Contains(t, params, "url")
	assert.Contains(t, params, "containerName")
	assert.Contains(t, params, "phase")
	assert.Contains(t, params, "dataRootDir")
	assert.Contains(t, params, "cgroupDriver")
}

func TestGenerateKubeletConfigMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return "/nonexistent/path/config.yaml"
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"dataRootDir": testDataRootDir,
	}

	err := kp.generateKubeletConfig(config)

	assert.Error(t, err)
}

func TestRenderKubeletServiceUnknownRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return testConfigPath
	})

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return "unknown"
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"imageRepo": testImageRepo,
	}

	err := kp.renderKubeletService(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown container runtime type")
}

func TestGenerateKubeletConfigByHostOSCentos(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "centos", "", "", nil
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"cgroupDriver": "systemd",
	}

	err := kp.generateKubeletConfigByHostOS(config)

	assert.NoError(t, err)
	assert.Equal(t, "systemd", config["cgroupDriver"])
}

func TestGenerateKubeletConfigByHostOSKylinWithDockerCE(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "kylin", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return nil
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"cgroupDriver": "systemd",
	}

	err := kp.generateKubeletConfigByHostOS(config)

	assert.NoError(t, err)
	assert.Equal(t, "systemd", config["cgroupDriver"])
}

func TestGenerateKubeletConfigByHostOSKylinWithoutDockerCE(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "kylin", "", "", nil
	})

	patches.ApplyFunc(httprepo.RepoSearch, func(repo string) error {
		return errors.New("repo not found")
	})

	patches.ApplyFunc((*kubeletPlugin).generateKubeletConfig, func(_ *kubeletPlugin, config map[string]string) error {
		return nil
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"cgroupDriver": "systemd",
	}

	err := kp.generateKubeletConfigByHostOS(config)

	assert.NoError(t, err)
	assert.Equal(t, "cgroupfs", config["cgroupDriver"])
}

func TestGenerateKubeletConfigByHostOSGetPlatformError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(host.PlatformInformation, func() (string, string, string, error) {
		return "", "", "", errors.New("failed to get platform")
	})

	kp := &kubeletPlugin{}

	config := map[string]string{}

	err := kp.generateKubeletConfigByHostOS(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get host platform info failed")
}

func TestExtraVolumesControlPlane(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"extraVolumes":    "",
		"manifestDir":     "/etc/kubernetes/manifests",
		"certificatesDir": "/etc/kubernetes/pki",
		"phase":           "ControlPlane",
	}

	result := kp.extraVolumes(config)

	assert.Contains(t, result, "/etc/kubernetes/manifests")
	assert.Contains(t, result, "/etc/kubernetes/pki")
}

func TestExtraVolumesWorker(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"extraVolumes":    "",
		"manifestDir":     "",
		"certificatesDir": "/etc/kubernetes/pki",
		"phase":           "Worker",
	}

	result := kp.extraVolumes(config)

	assert.NotContains(t, result, "/etc/kubernetes/manifests")
	assert.Contains(t, result, "/etc/kubernetes/pki")
}

func TestExtraVolumesWithExisting(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"extraVolumes":    "/data:/data",
		"manifestDir":     "",
		"certificatesDir": "/etc/kubernetes/pki",
		"phase":           "Worker",
	}

	result := kp.extraVolumes(config)

	assert.Contains(t, result, "/data:/data")
	assert.Contains(t, result, "/etc/kubernetes/pki")
}

func TestHandlerKubeletServiceParam(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return testConfigPath
	})

	kp := &kubeletPlugin{}

	config := map[string]string{
		"imageRepo": testImageRepo + "/",
		"extraArgs": "arg1;arg2",
		"hostIP":    testHostIP,
	}

	param := kp.handlerKubeletServiceParam(config)

	assert.Equal(t, filepath.FromSlash(testConfigPath)+"  ", param["kubeletConfig"])
	assert.Equal(t, testHostIP, param["hostIP"])
	assert.Equal(t, "test-node", param["hostName"])
	assert.Contains(t, param["podInfraContainerImage"], testImageRepo)
	assert.Contains(t, param["extraArgs"], "arg1")
	assert.Contains(t, param["extraArgs"], "arg2")
}

func TestGetProviderIDLineSuccess(t *testing.T) {
	config := map[string]string{
		"providerID": testProviderID,
	}

	line, err := getProviderIDLine(config)

	assert.NoError(t, err)
	assert.Equal(t, "providerID: "+testProviderID+"\n", line)
}

func TestGetProviderIDLineEmpty(t *testing.T) {
	config := map[string]string{
		"providerID": "",
	}

	line, err := getProviderIDLine(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty or missing")
	assert.Empty(t, line)
}

func TestGetProviderIDLineMissing(t *testing.T) {
	config := map[string]string{}

	line, err := getProviderIDLine(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty or missing")
	assert.Empty(t, line)
}

func TestEnsureDirExistsAlreadyExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, nil
	})

	err := ensureDirExists("/etc/kubernetes/config.yaml")

	assert.NoError(t, err)
}

func TestEnsureDirExistsCreatesDir(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	err := ensureDirExists("/nonexistent/dir/config.yaml")

	assert.NoError(t, err)
}

func TestEnsureDirExistsMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	err := ensureDirExists("/nonexistent/dir/config.yaml")

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
}

func TestOpenConfFile(t *testing.T) {
	tempFile := filepath.Join(t.TempDir(), "test.conf")

	file, err := openConfFile(tempFile)

	assert.NoError(t, err)
	assert.NotNil(t, file)
	file.Close()
}

func TestAppendProviderIDToConfYamlEmptyProviderID(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"providerID": "",
	}

	err := kp.appendProviderIDToConfYaml(config)

	assert.Error(t, err)
}

func TestDownloadAndInstallKubeletBinarySuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempDir := t.TempDir()

	patches.ApplyFunc(download.ExecDownload, func(url, saveto, rename, chmod string) error {
		return nil
	})

	kp := &kubeletPlugin{
		exec: &testExecutor{},
	}

	config := map[string]string{
		"url":    "http://example.com/kubelet",
		"saveto": tempDir,
		"rename": "kubelet",
		"chmod":  "0755",
	}

	err := kp.downloadAndInstallKubeletBinary(config)

	assert.NoError(t, err)
}

func TestDownloadAndInstallKubeletBinaryDownloadError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(download.ExecDownload, func(url, saveto, rename, chmod string) error {
		return errors.New("download failed")
	})

	kp := &kubeletPlugin{
		exec: &testExecutor{},
	}

	config := map[string]string{
		"url":    "http://example.com/kubelet",
		"saveto": "/nonexistent",
		"rename": "kubelet",
		"chmod":  "0755",
	}

	err := kp.downloadAndInstallKubeletBinary(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "download failed")
}

func TestJoinControlPlanePrepare(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"generateKubeletConfig": "true",
	}

	err := kp.joinControlPlanePrepare(config)

	assert.NoError(t, err)
}

func TestJoinWorkerPrepareGenerateConfigTrue(t *testing.T) {
	kp := &kubeletPlugin{}

	config := map[string]string{
		"generateKubeletConfig": "true",
	}

	err := kp.joinWorkerPrepare(config)

	assert.NoError(t, err)
}

func TestJoinWorkerPrepareNoK8sClient(t *testing.T) {
	kp := &kubeletPlugin{
		k8sClient: nil,
	}

	config := map[string]string{
		"generateKubeletConfig": "false",
	}

	err := kp.joinWorkerPrepare(config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no manager kubernetes cluster client")
}

func TestStoreKubeletConfigSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return nil
	})

	configMap := &corev1.ConfigMap{
		Data: map[string]string{
			"kubelet": "apiVersion: v1\nkind: KubeletConfiguration",
		},
	}

	err := storeKubeletConfig(configMap)

	assert.NoError(t, err)
}

func TestStoreKubeletConfigNoData(t *testing.T) {
	configMap := &corev1.ConfigMap{
		Data: map[string]string{},
	}

	err := storeKubeletConfig(configMap)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubelet config data not found")
}

func TestVariableSubstitutorBasic(t *testing.T) {
	exec := &testExecutor{}
	substitutor := &VariableSubstitutor{
		config: map[string]string{
			"VAR1": "value1",
			"VAR2": "value2",
		},
		exec: exec,
	}

	content := "test ${VAR1} and ${VAR2}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test value1 and value2", result)
}

func TestVariableSubstitutorEnvVar(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	os.Setenv("TEST_ENV_VAR", "env_value")
	defer os.Unsetenv("TEST_ENV_VAR")

	exec := &testExecutor{}
	substitutor := &VariableSubstitutor{
		config: map[string]string{
			"VAR1": "value1",
		},
		exec: exec,
	}

	content := "test ${VAR1} and ${TEST_ENV_VAR}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test value1 and env_value", result)
}

func TestVariableSubstitutorUnknownVar(t *testing.T) {
	exec := &testExecutor{}
	substitutor := &VariableSubstitutor{
		config: map[string]string{},
		exec:   exec,
	}

	content := "test ${UNKNOWN_VAR}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test ${UNKNOWN_VAR}", result)
}

func TestVariableSubstitutorEXPRCommand(t *testing.T) {
	exec := &testExecutor{
		output: "expr_result",
	}
	substitutor := &VariableSubstitutor{
		config: map[string]string{},
		exec:   exec,
	}

	content := "test ${EXPR|echo expr_result|END}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test expr_result", result)
}

func TestVariableSubstitutorEXPRCommandError(t *testing.T) {
	exec := &testExecutor{
		shouldFail:  true,
		failCommand: "echo",
	}
	substitutor := &VariableSubstitutor{
		config: map[string]string{},
		exec:   exec,
	}

	content := "test ${EXPR|echo test|END}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test ${EXPR|echo test|END}", result)
}

func TestVariableSubstitutorEmptyEXPR(t *testing.T) {
	exec := &testExecutor{}
	substitutor := &VariableSubstitutor{
		config: map[string]string{},
		exec:   exec,
	}

	content := "test ${EXPR||END}"

	result, err := substitutor.Substitute(content)

	assert.NoError(t, err)
	assert.Equal(t, "test ${EXPR||END}", result)
}

func TestProcessKubeletServiceNil(t *testing.T) {
	kp := &kubeletPlugin{}

	err := kp.processKubeletService(nil, map[string]string{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubelet.service is nil")
}

func TestGenerateServiceNewServiceDataNil(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(NewServiceData, func(configPath string) *ServiceGenerator {
		return nil
	})

	err := generateService(nil, map[string]string{}, nil)

	assert.Error(t, err)
}

func TestMountList(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	mounts := mountList()

	assert.NotEmpty(t, mounts)
	assert.True(t, len(mounts) > numTen)
}

func TestReadConfigFromKubeletConfigCRMissingName(t *testing.T) {
	kp := &kubeletPlugin{}

	err := kp.readConfigFromKubeletConfigCR(map[string]string{
		"useDeliveredConfig": "true",
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubeletConfigName is required")
}

const (
	numEleven     = 11
	numTwelve     = 12
	numNinetyNine = 99

	shortSleepDuration = 10 * time.Millisecond
	longSleepDuration  = 100 * time.Millisecond

	testSleepTime       = 1
	testRawConfig       = `{"raw":"apiVersion: v1\nkind: KubeletConfiguration"}`
	testKubeletActive   = "active"
	testKubeletInactive = "inactive"
)

type mockFileInfo struct {
	size int64
}

func (m *mockFileInfo) Name() string       { return "test" }
func (m *mockFileInfo) Size() int64        { return m.size }
func (m *mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m *mockFileInfo) ModTime() time.Time { return time.Now() }
func (m *mockFileInfo) IsDir() bool        { return false }
func (m *mockFileInfo) Sys() interface{}   { return nil }

func TestEnsureFileEndsWithNewlineEmptyFile(t *testing.T) {
	tempFile := filepath.Join(t.TempDir(), "empty.txt")
	file, err := os.Create(tempFile)
	assert.NoError(t, err)
	defer file.Close()

	err = ensureFileEndsWithNewline(file)

	assert.NoError(t, err)
}

func TestEnsureFileEndsWithNewlineWithNewline(t *testing.T) {
	tempFile := filepath.Join(t.TempDir(), "with_newline.txt")
	file, err := os.Create(tempFile)
	assert.NoError(t, err)
	defer file.Close()

	_, err = file.WriteString("test content\n")
	assert.NoError(t, err)

	err = ensureFileEndsWithNewline(file)

	assert.NoError(t, err)
}

func TestEnsureFileEndsWithNewlineWithoutNewline(t *testing.T) {
	tempFile := filepath.Join(t.TempDir(), "without_newline.txt")
	file, err := os.Create(tempFile)
	assert.NoError(t, err)
	defer file.Close()

	_, err = file.WriteString("test content")
	assert.NoError(t, err)

	err = ensureFileEndsWithNewline(file)

	assert.NoError(t, err)

	content, err := os.ReadFile(tempFile)
	assert.NoError(t, err)
	assert.True(t, len(content) > numEleven)
	assert.Equal(t, byte('\n'), content[len(content)-numOne])
}

func TestProcessKubeletConfigurationUnmarshalError(t *testing.T) {
	kp := &kubeletPlugin{}

	kubeletConfig := map[string]Runtime.RawExtension{
		"kubelet.conf": {
			Raw: []byte("invalid json"),
		},
	}

	config := map[string]string{}

	err := kp.processKubeletConfiguration(kubeletConfig, config)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to unmarshal raw config")
}

func TestNewKubeletScriptSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return "/var/lib/kubelet/kubelet.conf"
	})

	patches.ApplyFunc(pkiutil.GetDefaultKubeConfigPath, func() string {
		return "/etc/kubernetes/kubelet.conf"
	})

	patches.ApplyFunc(mfutil.GetDefaultManifestsPath, func() string {
		return "/etc/kubernetes/manifests"
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	config := map[string]string{
		"kubeletImage":      "test-image:v1.0",
		"pauseImage":        "pause-image:v3.0",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": "v1.21.0",
		"hostIP":            "192.168.1.100",
		"dataRootDir":       "/var/lib/kubelet",
		"phase":             "ControlPlane",
	}

	err := newKubeletScript(config)

	assert.NoError(t, err)
}

func TestNewKubeletScriptWithEmptyConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return "/var/lib/kubelet/kubelet.conf"
	})

	patches.ApplyFunc(pkiutil.GetDefaultKubeConfigPath, func() string {
		return "/etc/kubernetes/kubelet.conf"
	})

	patches.ApplyFunc(mfutil.GetDefaultManifestsPath, func() string {
		return "/etc/kubernetes/manifests"
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	config := map[string]string{
		"kubeletImage":      "test-image:v1.0",
		"pauseImage":        "pause-image:v3.0",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": "v1.21.0",
		"hostIP":            "192.168.1.100",
		"dataRootDir":       "/var/lib/kubelet",
		"phase":             "Worker",
	}

	err := newKubeletScript(config)

	assert.NoError(t, err)
}

func TestNewKubeletScriptExportPhase(t *testing.T) {
	t.Skip("skip unstable UT: writes kubelet script under /etc/kubernetes and requires host permissions")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletConfPath, func() string {
		return "/var/lib/kubelet/kubelet.conf"
	})

	patches.ApplyFunc(pkiutil.GetDefaultKubeConfigPath, func() string {
		return "/etc/kubernetes/kubelet.conf"
	})

	patches.ApplyFunc(mfutil.GetDefaultManifestsPath, func() string {
		return "/etc/kubernetes/manifests"
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	config := map[string]string{
		"kubeletImage":      "test-image:v1.0",
		"pauseImage":        "pause-image:v3.0",
		"kubeconfigPath":    "/etc/kubernetes/kubelet.conf",
		"extraVolumes":      "",
		"extraArgs":         "",
		"kubernetesVersion": "v1.21.0",
		"hostIP":            "192.168.1.100",
		"dataRootDir":       "/var/lib/kubelet",
		"phase":             "UpgradeControlPlane",
	}

	err := newKubeletScript(config)

	assert.NoError(t, err)
}
