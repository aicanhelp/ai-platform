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
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	edocker "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/download"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
)

type mockExecutor struct {
	output string
	err    error
}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return m.err
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return m.err
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return m.err
}

func patchDockerHostDeps(t *testing.T) {
	patches := gomonkey.NewPatches()
	t.Cleanup(patches.Reset)
	patches.ApplyFunc(httprepo.RepoSearch, func(pkg string) error {
		return nil
	})
	patches.ApplyFunc(httprepo.RepoInstall, func(pkg ...string) error {
		return nil
	})
	patches.ApplyFunc(edocker.ConfigDockerDaemon, func(params edocker.DockerDaemonConfig) error {
		return nil
	})
	patches.ApplyFunc(edocker.WaitDockerReady, func() error {
		return nil
	})
	patches.ApplyFunc(download.ExecDownload, func(url, saveto, rename, chmod string) error {
		return nil
	})
}

func TestDockerPluginName(t *testing.T) {
	plugin := &DockerPlugin{}
	assert.Equal(t, Name, plugin.Name())
}

func TestDockerPluginParam(t *testing.T) {
	plugin := &DockerPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "insecureRegistries")
	assert.Contains(t, params, "cgroupDriver")
	assert.Contains(t, params, "runtime")
	assert.Contains(t, params, "dataRoot")
	assert.Contains(t, params, "enableDockerTls")
	assert.Contains(t, params, "tlsHost")
	assert.Contains(t, params, "runtimeUrl")
	assert.Contains(t, params, "registryMirror")
}

func TestDockerPluginParamDefaults(t *testing.T) {
	plugin := &DockerPlugin{}
	params := plugin.Param()
	assert.Equal(t, defaultCgroupDriver, params["cgroupDriver"].Default)
	assert.Equal(t, defaultRuntime, params["runtime"].Default)
	assert.Equal(t, defaultDataRoot, params["dataRoot"].Default)
	assert.Equal(t, "false", params["enableDockerTls"].Default)
}

func TestNewDockerPlugin(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "systemd", defaultCgroupDriver)
	assert.Equal(t, "runc", defaultRuntime)
	assert.Equal(t, "/var/lib/docker", defaultDataRoot)
}

func TestDockerPluginFields(t *testing.T) {
	plugin := &DockerPlugin{}
	assert.NotNil(t, plugin)
}

func TestDockerPluginParamRequired(t *testing.T) {
	plugin := &DockerPlugin{}
	params := plugin.Param()
	assert.False(t, params["insecureRegistries"].Required)
	assert.False(t, params["registryMirror"].Required)
	assert.False(t, params["cgroupDriver"].Required)
	assert.False(t, params["runtime"].Required)
	assert.False(t, params["dataRoot"].Required)
	assert.False(t, params["enableDockerTls"].Required)
	assert.False(t, params["tlsHost"].Required)
	assert.False(t, params["runtimeUrl"].Required)
}

func TestDockerPluginParamDescriptions(t *testing.T) {
	plugin := &DockerPlugin{}
	params := plugin.Param()
	for key, param := range params {
		assert.NotEmpty(t, param.Description, "Description for %s should not be empty", key)
	}
}

func TestDockerPluginWithMockExecutor(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: nil}
	plugin := New(mockExec)
	assert.NotNil(t, plugin)
}

func TestDockerPluginExecuteWithDefaultParams(t *testing.T) {
	patchDockerHostDeps(t)
	mockExec := &mockExecutor{output: "", err: nil}
	plugin := New(mockExec)
	commands := []string{Name}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestDockerPluginExecuteWithRichRunc(t *testing.T) {
	patchDockerHostDeps(t)
	mockExec := &mockExecutor{output: "", err: nil}
	plugin := New(mockExec)
	commands := []string{Name, "runtime=richrunc", "runtimeUrl=http://example.com/runc"}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestDockerPluginExecuteWithAllParams(t *testing.T) {
	patchDockerHostDeps(t)
	mockExec := &mockExecutor{output: "", err: nil}
	plugin := New(mockExec)
	commands := []string{
		Name,
		"insecureRegistries=registry.example.com",
		"registryMirror=mirror.example.com",
		"cgroupDriver=systemd",
		"runtime=runc",
		"dataRoot=/var/lib/docker",
		"enableDockerTls=false",
		"tlsHost=",
	}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestDockerPluginExecuteWithEnableTls(t *testing.T) {
	patchDockerHostDeps(t)
	mockExec := &mockExecutor{output: "", err: nil}
	plugin := New(mockExec)
	commands := []string{
		Name,
		"enableDockerTls=true",
		"tlsHost=192.168.1.1",
	}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestDockerPluginExecuteStartFailure(t *testing.T) {
	patchDockerHostDeps(t)
	mockExec := &mockExecutor{output: "failed to start", err: fmt.Errorf("start failed")}
	plugin := New(mockExec)
	commands := []string{Name}
	_, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
}
