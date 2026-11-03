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

package selfupdate

import (
	"context"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type mockSelfUpdateExecutor struct {
	exec.Executor
	executeCommandCalled     bool
	executeCommandWithOutput string
	executeCommandError      error
	executeCommandArgs       []string
}

func (m *mockSelfUpdateExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockSelfUpdateExecutor) ExecuteCommand(_ string, args ...string) error {
	m.executeCommandCalled = true
	m.executeCommandArgs = append([]string(nil), args...)
	return m.executeCommandError
}

func (m *mockSelfUpdateExecutor) ExecuteCommandWithCombinedOutput(_ string, _ ...string) (string, error) {
	m.executeCommandCalled = true
	return m.executeCommandWithOutput, m.executeCommandError
}

func (m *mockSelfUpdateExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func patchDownloadAgentBinary(patches *gomonkey.Patches) {
	patches.ApplyFunc(downloadAgentBinary, func(_ string) (string, error) {
		return path.Join(utils.AgentBin, agentBinaryName), nil
	})
}

func TestUpdatePluginName(t *testing.T) {
	pluginObj := &UpdatePlugin{}
	assert.Equal(t, Name, pluginObj.Name())
}

func TestUpdatePluginParam(t *testing.T) {
	pluginObj := &UpdatePlugin{}
	params := pluginObj.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "agentUrl")
	assert.Equal(t, DefaultAgentUrl, params["agentUrl"].Default)
	assert.Equal(t, "Agent download url used to replace existing BKEAgent binary", params["agentUrl"].Description)
}

func TestNewUpdatePlugin(t *testing.T) {
	mockExec := &mockSelfUpdateExecutor{}
	pluginObj := New(mockExec)
	assert.NotNil(t, pluginObj)
	assert.Equal(t, Name, pluginObj.Name())
}

func TestNeedUpdateWhenBinNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return false
	})

	mockExec := &mockSelfUpdateExecutor{}
	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.True(t, result)
	assert.False(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenBinExistsCommandFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = ""
	mockExec.executeCommandError = errors.New("command failed")

	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.True(t, result)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenBinExistsSuccessWithDifferentVersion(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.True(t, result)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenBinExistsSuccessWithSameVersion(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "dev"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.False(t, result)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenBinExistsOutputEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = ""
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.True(t, result)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenBinExistsOutputHasMultipleLines(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123\nSome other output\nMore lines"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.True(t, result)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestNeedUpdateWhenPrintVersionOutputSameCommit(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "🤯 Version: latest\n🤔 GitCommitId: dev \n👉 Architecture: unknown\n"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	assert.False(t, pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent"))
}

func TestNeedUpdateWhenPrintVersionOutputDifferentCommit(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "🤯 Version: latest\n🤔 GitCommitId: 3333333333 \n"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	assert.True(t, pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent"))
}

func TestGitCommitFromVersionOutput(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   string
	}{
		{
			name: "print version",
			output: "🤯 Version: latest\n" +
				"🤔 GitCommitId: 3333333333 \n" +
				"👉 Architecture: unknown\n",
			want: "3333333333",
		},
		{"single line legacy", "dev", "dev"},
		{"no commit line", "🤯 Version: latest\n", ""},
		{"empty", "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, gitCommitFromVersionOutput(tc.output))
		})
	}
}

func TestExecuteWhenNoUpdateNeeded(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "dev"

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteWhenScriptsDirNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	var mkdirCalled bool
	var writeFileCalled bool

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		if strings.Contains(s, "scripts") && !strings.Contains(s, "update.sh") {
			return false
		}
		if strings.Contains(s, "update.sh") {
			return false
		}
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(s string, _ os.FileMode) error {
		mkdirCalled = true
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, _ []byte, _ os.FileMode) error {
		writeFileCalled = true
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "different-version"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, mkdirCalled)
	assert.True(t, writeFileCalled)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestExecuteWhenScriptsDirExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	var mkdirCalled bool
	var writeFileCalled bool

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		if s == utils.AgentScripts {
			return true
		}
		if s == restartScript {
			return false
		}
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(s string, _ os.FileMode) error {
		if s == utils.AgentScripts {
			mkdirCalled = true
		}
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(s string, _ []byte, _ os.FileMode) error {
		if s == restartScript {
			writeFileCalled = true
		}
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "different-version"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.False(t, mkdirCalled)
	assert.True(t, writeFileCalled)
}

func TestExecuteWhenScriptExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	var mkdirCalled bool
	var writeFileCalled bool

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		if s == utils.AgentScripts || s == restartScript {
			return true
		}
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(s string, _ os.FileMode) error {
		if s == utils.AgentScripts {
			mkdirCalled = true
		}
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(s string, _ []byte, _ os.FileMode) error {
		if s == restartScript {
			writeFileCalled = true
		}
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.False(t, mkdirCalled)
	assert.False(t, writeFileCalled)
}

func TestExecuteWithMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	mkdirErr := errors.New("mkdir failed")

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return mkdirErr
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "different-version"

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
	assert.Nil(t, result)
}

func TestExecuteWithWriteFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	writeErr := errors.New("write file failed")

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		if s == utils.AgentScripts {
			return true
		}
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, _ []byte, _ os.FileMode) error {
		return writeErr
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "different-version"

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "write file failed")
	assert.Nil(t, result)
}

func TestExecuteWithExecuteCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	execErr := errors.New("execute command failed")

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, _ []byte, _ os.FileMode) error {
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"
	mockExec.executeCommandError = execErr

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execute command failed")
	assert.Nil(t, result)
}

func TestExecuteWithDownloadError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(downloadAgentBinary, func(_ string) (string, error) {
		return "", errors.New("download failed")
	})

	pluginObj := &UpdatePlugin{exec: &mockSelfUpdateExecutor{}}
	result, err := pluginObj.Execute(context.Background(), []string{Name, "agentUrl=http://test.com/bkeagent"})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "download failed")
	assert.Nil(t, result)
}

func TestExecuteWithCustomAgentUrl(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	customUrl := "http://custom.example.com/bkeagent-latest-linux-{.arch}"
	var downloadedURL string
	patches.ApplyFunc(downloadAgentBinary, func(url string) (string, error) {
		downloadedURL = url
		return path.Join(utils.AgentBin, agentBinaryName), nil
	})

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, _ []byte, _ os.FileMode) error {
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"
	mockExec.executeCommandError = nil

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name, "agentUrl=" + customUrl})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.Equal(t, customUrl, downloadedURL)
}

func TestUpdatePluginDefaultAgentUrl(t *testing.T) {
	pluginObj := &UpdatePlugin{}
	params := pluginObj.Param()

	assert.NotNil(t, params["agentUrl"])
	assert.Equal(t, DefaultAgentUrl, params["agentUrl"].Value)
	assert.Equal(t, DefaultAgentUrl, params["agentUrl"].Default)
	assert.False(t, params["agentUrl"].Required)
}

func TestUpdatePluginConstantName(t *testing.T) {
	assert.Equal(t, "SelfUpdate", Name)
	assert.Equal(t, "http://http.bocloud.k8s:40080/files/bkeagent-latest-linux-{.arch}", DefaultAgentUrl)
}

func TestUpdatePluginRestartScriptPath(t *testing.T) {
	scriptPath := path.Join(utils.AgentScripts, "update.sh")
	assert.Equal(t, restartScript, scriptPath)
}

func TestUpdatePluginNeedUpdateVersionOutputCases(t *testing.T) {
	testCases := []struct {
		name           string
		output         string
		expectedResult bool
	}{
		{"single line different commit", "abc123", true},
		{"version banner same commit", "🤯 Version: latest\n🤔 GitCommitId: dev \n", false},
		{"version banner different commit", "🤯 Version: latest\n🤔 GitCommitId: other \n", true},
		{"empty output", "", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(_ string) bool {
				return true
			})

			mockExec := &mockSelfUpdateExecutor{}
			mockExec.executeCommandWithOutput = tc.output

			pluginObj := &UpdatePlugin{exec: mockExec}

			result := pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func TestUpdatePluginExecuteParsesCommands(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	patches.ApplyFunc(utils.Exists, func(_ string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, _ []byte, _ os.FileMode) error {
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"

	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name, "agentUrl=http://test.com/bkeagent"})

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestUpdatePluginNeedUpdateBinaryPathConstruction(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	var binaryPath string

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		binaryPath = s
		return true
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "dev"

	pluginObj := &UpdatePlugin{exec: mockExec}
	pluginObj.NeedUpdate("/etc/bkeagent/bin/bkeagent")

	assert.Equal(t, "/etc/bkeagent/bin/bkeagent", binaryPath)
}

func TestUpdatePluginUpdateScriptContent(t *testing.T) {
	assert.NotEmpty(t, updateScript)
	assert.True(t, strings.HasPrefix(updateScript, "#!/bin/bash"))
	assert.Contains(t, updateScript, "set -euo pipefail")
}

func TestUpdatePluginCommandsArrayParsing(t *testing.T) {
	testCases := []struct {
		name        string
		commands    []string
		expectError bool
	}{
		{"empty commands", []string{}, false},
		{"plugin name only", []string{Name}, false},
		{"with agent url", []string{Name, "agentUrl=http://test.com"}, false},
		{"multiple commands", []string{Name, "agentUrl=http://test.com", "extra=value"}, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patchDownloadAgentBinary(patches)

			patches.ApplyFunc(utils.Exists, func(_ string) bool {
				return true
			})

			mockExec := &mockSelfUpdateExecutor{}
			mockExec.executeCommandWithOutput = "dev"

			pluginObj := &UpdatePlugin{exec: mockExec}

			commands := tc.commands
			if len(commands) == 1 && commands[0] == Name {
				commands = []string{Name, "agentUrl=http://test.com/bkeagent"}
			}
			result, err := pluginObj.Execute(context.Background(), commands)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Nil(t, result)
			}
		})
	}
}

func TestExecuteSchedulesDelayedRestartCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	patches.ApplyFunc(utils.Exists, func(_ string) bool { return true })

	mockExec := &mockSelfUpdateExecutor{
		executeCommandWithOutput: "different-version",
	}
	pluginObj := &UpdatePlugin{exec: mockExec}

	result, err := pluginObj.Execute(context.Background(), []string{Name, "agentUrl=http://test.com/bkeagent"})

	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.True(t, mockExec.executeCommandCalled)
	assert.Equal(t, []string{"-c", "nohup /bin/sh -c 'sleep 3; /etc/openFuyao/bkeagent/scripts/update.sh /etc/openFuyao/bkeagent/bin/bkeagent' >/dev/null 2>&1 &"}, mockExec.executeCommandArgs)
}

func TestUpdatePluginScriptsDirectoryConstant(t *testing.T) {
	expected := "/etc/openFuyao/bkeagent/scripts"
	actual := utils.AgentScripts
	assert.Equal(t, expected, actual)
}

func TestUpdatePluginBinDirectoryConstant(t *testing.T) {
	expected := "/etc/openFuyao/bkeagent/bin"
	actual := utils.AgentBin
	assert.Equal(t, expected, actual)
}

func TestUpdatePluginUpdateScriptPermission(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchDownloadAgentBinary(patches)

	patches.ApplyFunc(utils.Exists, func(s string) bool {
		if s == utils.AgentScripts {
			return true
		}
		return false
	})

	patches.ApplyFunc(os.WriteFile, func(_ string, content []byte, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockSelfUpdateExecutor{}
	mockExec.executeCommandWithOutput = "abc123"

	pluginObj := &UpdatePlugin{exec: mockExec}
	pluginObj.Execute(context.Background(), []string{Name})

}
