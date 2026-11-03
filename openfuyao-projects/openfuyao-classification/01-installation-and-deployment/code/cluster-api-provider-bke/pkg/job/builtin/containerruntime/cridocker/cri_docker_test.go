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

package cridocker

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

const (
	numZero  = 0
	numOne   = 1
	numTwo   = 2
	numThree = 3
	numFour  = 4
)

const (
	testSandboxImage  = "test.registry.io/kubernetes/pause:3.8"
	testCriDockerdURL = "http://test.com/cri-dockerd"
)

const (
	shortWaitDuration = 100 * time.Millisecond
	longWaitDuration  = 10 * time.Second
)

type mockExecutorForExecute struct{}

func (m *mockExecutorForExecute) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutorForExecute) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "daemon-reload") {
		return "success", nil
	}
	if strings.Contains(command, "enable cri-dockerd") {
		return "success", nil
	}
	if strings.Contains(command, "restart cri-dockerd") {
		return "success", nil
	}
	return "success", nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorForExecute) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorForExecute) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorForExecute) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
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
	return "success", nil
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

func newMockExecutor() *mockExecutor {
	return &mockExecutor{}
}

func newMockExecutorForExecute() *mockExecutorForExecute {
	return &mockExecutorForExecute{}
}

func TestCRIDockerPluginName(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	assert.Equal(t, Name, plugin.Name())
}

func TestCRIDockerPluginParam(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "criDockerdUrl")
	assert.Contains(t, params, "sandbox")
}

func TestCRIDockerPluginParamDefaults(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	assert.Equal(t, defaultSandbox, params["sandbox"].Default)
}

func TestCRIDockerPluginParamRequired(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	assert.True(t, params["criDockerdUrl"].Required)
	assert.True(t, params["sandbox"].Required)
}

func TestNewCRIDockerPlugin(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestNewCRIDockerPluginWithExecutor(t *testing.T) {
	executor := newMockExecutor()
	plugin := New(executor)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestStartCriDockerdSuccess(t *testing.T) {
	executor := newMockExecutor()
	plugin := &CRIDockerPlugin{exec: executor}
	err := plugin.startCriDockerd()
	assert.NoError(t, err)
}

func TestExecuteParseCommandsError(t *testing.T) {
	p := &CRIDockerPlugin{exec: newMockExecutor()}
	commands := []string{Name, "criDockerdUrl=" + testCriDockerdURL}

	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestCRIDockerPluginWithNilExecutor(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
}

func TestCRIDockerPluginParamContainsAll(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	assert.Equal(t, numTwo, len(params))
	assert.Contains(t, params, "criDockerdUrl")
	assert.Contains(t, params, "sandbox")
}

func TestCRIDockerPluginParamDescriptions(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	for key, param := range params {
		assert.NotEmpty(t, param.Description, "Description for %s should not be empty", key)
	}
}

func TestStartCriDockerdWithNilExecutor(t *testing.T) {
	plugin := &CRIDockerPlugin{exec: nil}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Expected panic due to nil exec: %v", r)
		}
	}()
	_ = plugin.startCriDockerd()
}

func TestExecuteWithStartError(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cri-dockerd-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	servicePath := filepath.Join(tempDir, "cri-dockerd.service")
	socketPath := filepath.Join(tempDir, "cri-dockerd.socket")

	f1, err := os.Create(servicePath)
	assert.NoError(t, err)
	f1.Close()

	f2, err := os.Create(socketPath)
	assert.NoError(t, err)
	f2.Close()

	mockExec := &mockExecutorWithRestartError{}
	plugin := &CRIDockerPlugin{exec: mockExec}
	commands := []string{
		"InstallCRIDocker",
		"criDockerdUrl=" + testCriDockerdURL,
		"sandbox=" + testSandboxImage,
	}

	_, err = plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
}

type mockExecutorWithRestartError struct{}

func (m *mockExecutorWithRestartError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutorWithRestartError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "restart cri-dockerd") {
		return "", assert.AnError
	}
	return "success", nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithRestartError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestNewWithNilExecutor(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestParamMapNotNil(t *testing.T) {
	plugin := &CRIDockerPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.GreaterOrEqual(t, len(params), numTwo)
}

func TestStartCriDockerdWithDaemonReloadError(t *testing.T) {
	mockExec := &mockExecutorWithDaemonReloadError{}
	plugin := &CRIDockerPlugin{exec: mockExec}

	err := plugin.startCriDockerd()
	assert.NoError(t, err)
}

type mockExecutorWithDaemonReloadError struct{}

func (m *mockExecutorWithDaemonReloadError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "daemon-reload") {
		return "", assert.AnError
	}
	return "success", nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithDaemonReloadError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestStartCriDockerdWithEnableError(t *testing.T) {
	mockExec := &mockExecutorWithEnableError{}
	plugin := &CRIDockerPlugin{exec: mockExec}

	err := plugin.startCriDockerd()
	assert.NoError(t, err)
}

type mockExecutorWithEnableError struct{}

func (m *mockExecutorWithEnableError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutorWithEnableError) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if strings.Contains(command, "enable cri-dockerd") {
		return "", assert.AnError
	}
	return "success", nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithEnableError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

type mockExecutorWithStopError struct{}

func (m *mockExecutorWithStopError) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutorWithStopError) ExecuteCommand(command string, arg ...string) error {
	if strings.Contains(command, "systemctl stop") {
		return assert.AnError
	}
	return nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorWithStopError) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithStopError) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (m *mockExecutorWithStopError) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestStartCriDockerdWithStopError(t *testing.T) {
	mockExec := &mockExecutorWithStopError{}
	plugin := &CRIDockerPlugin{exec: mockExec}

	err := plugin.startCriDockerd()
	assert.NoError(t, err)
}

func TestWriteCriDockerdConfigToDiskOpenFileError(t *testing.T) {
	patches := gomonkey.ApplyFuncSeq(os.OpenFile, []gomonkey.OutputCell{
		{Values: gomonkey.Params{nil, assert.AnError}, Times: 1},
	})
	defer patches.Reset()

	runtimeParam := map[string]string{
		"sandbox": testSandboxImage,
	}

	err := writeCriDockerdConfigToDisk(runtimeParam)
	assert.Error(t, err)
}

func TestWriteCriDockerdConfigToDiskWriteSocketError(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cri-dockerd-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	servicePath := filepath.Join(tempDir, "cri-dockerd.service")

	serviceFile, err := os.Create(servicePath)
	assert.NoError(t, err)
	serviceFile.Close()

	patches := gomonkey.ApplyFuncSeq(os.OpenFile, []gomonkey.OutputCell{
		{Values: gomonkey.Params{serviceFile, nil}, Times: 1},
	})
	defer patches.Reset()

	patches.ApplyFuncSeq(os.WriteFile, []gomonkey.OutputCell{
		{Values: gomonkey.Params{assert.AnError}, Times: 1},
	})

	runtimeParam := map[string]string{
		"sandbox": testSandboxImage,
	}

	err = writeCriDockerdConfigToDisk(runtimeParam)
	assert.Error(t, err)
}

func TestWriteCriDockerdConfigToDiskTemplateExecuteError(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cri-dockerd-test")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	servicePath := filepath.Join(tempDir, "cri-dockerd.service")

	serviceFile, err := os.Create(servicePath)
	assert.NoError(t, err)
	serviceFile.Close()

	patches := gomonkey.ApplyFuncSeq(os.OpenFile, []gomonkey.OutputCell{
		{Values: gomonkey.Params{serviceFile, nil}, Times: 1},
	})
	defer patches.Reset()

	patches.ApplyFuncSeq(os.WriteFile, []gomonkey.OutputCell{
		{Values: gomonkey.Params{assert.AnError}, Times: 1},
	})

	runtimeParam := map[string]string{}

	err = writeCriDockerdConfigToDisk(runtimeParam)
	assert.Error(t, err)
}
