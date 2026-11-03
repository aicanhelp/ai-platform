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

package exec

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	numZero      = 0
	numOne       = 1
	numTwo       = 2
	numThree     = 3
	numFour      = 4
	numFive      = 5
	numTen       = 10
	shortWait    = 10 * time.Millisecond
	mediumWait   = 50 * time.Millisecond
	longWait     = 100 * time.Millisecond
	testTimeout  = 1 * time.Second
	shortTimeout = 100 * time.Millisecond
	testCommand  = "echo"
	testArg      = "hello"
	testOutput   = "test output"
	testError    = "test error"
	testPrefix   = "test"
)

type mockReadCloser struct {
	reader io.Reader
}

func (m *mockReadCloser) Read(p []byte) (n int, err error) {
	return m.reader.Read(p)
}

func (m *mockReadCloser) Close() error {
	return nil
}

func TestExecuteCommandSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		cmd := &exec.Cmd{}
		return StartCommandResult{Cmd: cmd}, nil
	})
	patches.ApplyFunc(logOutput, func(stdout, stderr io.ReadCloser) {})
	var waitCalled bool
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		waitCalled = true
		return nil
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommand(testCommand, testArg)

	assert.NoError(t, err)
	assert.True(t, waitCalled)
}

func TestExecuteCommandStartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	startError := os.ErrNotExist
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		return StartCommandResult{}, startError
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommand(testCommand, testArg)

	assert.Error(t, err)
	assert.Equal(t, startError, err)
}

func TestExecuteCommandWaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	waitError := os.ErrPermission
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		cmd := &exec.Cmd{}
		return StartCommandResult{
			Cmd:    cmd,
			Stdout: &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))},
			Stderr: &mockReadCloser{reader: bytes.NewReader([]byte(testError))},
		}, nil
	})
	patches.ApplyFunc(logOutput, func(stdout, stderr io.ReadCloser) {})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return waitError
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommand(testCommand, testArg)

	assert.Error(t, err)
	assert.Equal(t, waitError, err)
}

func TestExecuteCommandWithEnvSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		cmd := &exec.Cmd{}
		return StartCommandResult{Cmd: cmd}, nil
	})
	patches.ApplyFunc(logOutput, func(stdout, stderr io.ReadCloser) {})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	testEnv := []string{"TEST=value"}
	err := executor.ExecuteCommandWithEnv(testEnv, testCommand, testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandWithEnvEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		cmd := &exec.Cmd{}
		return StartCommandResult{Cmd: cmd}, nil
	})
	patches.ApplyFunc(logOutput, func(stdout, stderr io.ReadCloser) {})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommandWithEnv([]string{}, testCommand, testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandWithTimeoutSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithTimeout(testTimeout, testCommand, testArg)

	assert.NoError(t, err)
	assert.NotNil(t, output)
}

func TestExecuteCommandWithTimeoutStartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	startError := os.ErrNotExist
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return startError
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithTimeout(testTimeout, testCommand, testArg)

	assert.Error(t, err)
	assert.Equal(t, startError, err)
	assert.Equal(t, "", output)
}

func TestExecuteCommandWithOutputSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithOutput(testCommand, testArg)

	assert.NoError(t, err)
	assert.Equal(t, testOutput, output)
}

func TestExecuteCommandWithOutputError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	outputError := os.ErrPermission
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(testOutput), outputError
	})

	executor := &CommandExecutor{}
	_, err := executor.ExecuteCommandWithOutput(testCommand, testArg)

	assert.Error(t, err)
	assert.Equal(t, os.ErrPermission, err)
}

func TestExecuteCommandWithCombinedOutputSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput(testCommand, testArg)

	assert.NoError(t, err)
	assert.Equal(t, testOutput, output)
}

func TestExecuteCommandWithCombinedOutputError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	outputError := os.ErrPermission
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), outputError
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput(testCommand, testArg)

	assert.Error(t, err)
	assert.Equal(t, testOutput, output)
}

func TestExecuteCommandWithCombinedOutputInDirSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutputInDir("/tmp", testCommand, testArg)

	assert.NoError(t, err)
	assert.Equal(t, testOutput, output)
}

func TestExecuteCommandResidentBinary(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommandResidentBinary(shortWait, testCommand, testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandResidentBinaryRunError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return os.ErrPermission
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommandResidentBinary(shortWait, testCommand, testArg)

	assert.NoError(t, err)
}

func TestStartCommandSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testError))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})

	result, err := startCommand(context.Background(), []string{}, testCommand, testArg)

	assert.NoError(t, err)
	assert.NotNil(t, result.Cmd)
	assert.NotNil(t, result.Stdout)
	assert.NotNil(t, result.Stderr)
}

func TestStartCommandWithEnv(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testEnv := []string{"TEST=value"}
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testError))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})

	result, err := startCommand(context.Background(), testEnv, testCommand, testArg)

	assert.NoError(t, err)
	assert.NotNil(t, result.Cmd)
	assert.NotNil(t, result.Stdout)
	assert.NotNil(t, result.Stderr)
}

func TestStartCommandStdoutPipeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pipeError := os.ErrClosed
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return nil, pipeError
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testError))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})

	result, err := startCommand(context.Background(), []string{}, testCommand, testArg)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "open stdout pipe")
	assert.True(t, errors.Is(err, pipeError))
	assert.Nil(t, result.Cmd)
	assert.Nil(t, result.Stdout)
	assert.Nil(t, result.Stderr)
}

func TestStartCommandStderrPipeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pipeError := os.ErrClosed
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return nil, pipeError
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})

	result, err := startCommand(context.Background(), []string{}, testCommand, testArg)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "open stderr pipe")
	assert.True(t, errors.Is(err, pipeError))
	assert.Nil(t, result.Cmd)
	assert.Nil(t, result.Stdout)
	assert.Nil(t, result.Stderr)
}

func TestStartCommandStartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	startError := os.ErrPermission
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testError))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return startError
	})

	result, err := startCommand(context.Background(), []string{}, testCommand, testArg)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "start command")
	assert.True(t, errors.Is(err, startError))
	assert.Nil(t, result.Cmd)
	assert.Nil(t, result.Stdout)
	assert.Nil(t, result.Stderr)
}

func TestLogFromReader(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Debug, func(msg string, v ...interface{}) {})

	reader := &mockReadCloser{reader: strings.NewReader("line1\nline2\nline3")}
	logFromReader(reader)
}

func TestLogOutputBothNil(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Warnf, func(format string, v ...interface{}) {})

	logOutput(nil, nil)
}

func TestLogOutputStdoutNil(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Warnf, func(format string, v ...interface{}) {})

	logOutput(nil, &mockReadCloser{reader: strings.NewReader("error")})
}

func TestLogOutputStderrNil(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Warnf, func(format string, v ...interface{}) {})

	logOutput(&mockReadCloser{reader: strings.NewReader("output")}, nil)
}

func TestRunCommandWithOutputSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	output, err := runCommandWithOutput(&exec.Cmd{}, false)

	assert.NoError(t, err)
	assert.Equal(t, testOutput, output)
}

func TestRunCommandWithCombinedOutputSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	output, err := runCommandWithOutput(&exec.Cmd{}, true)

	assert.NoError(t, err)
	assert.Equal(t, testOutput, output)
}

func TestRunCommandWithOutputError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	outputError := os.ErrPermission
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(testOutput), outputError
	})

	_, err := runCommandWithOutput(&exec.Cmd{}, false)

	assert.Error(t, err)
	assert.Equal(t, os.ErrPermission, err)
}

func TestRunCommandWithCombinedOutputError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	outputError := os.ErrPermission
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), outputError
	})

	output, err := runCommandWithOutput(&exec.Cmd{}, true)

	assert.Error(t, err)
	assert.Equal(t, testOutput, output)
}

func TestLogCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Infof, func(format string, v ...interface{}) {})

	logCommand(testCommand, testArg)
}

func TestAssertErrorTypeExitError(t *testing.T) {
	exitError := &exec.ExitError{
		Stderr: []byte("exit error message"),
	}
	result := assertErrorType(exitError)

	assert.Equal(t, "exit error message", result)
}

func TestAssertErrorTypeError(t *testing.T) {
	execError := &exec.Error{
		Name: "test command",
		Err:  os.ErrNotExist,
	}
	result := assertErrorType(execError)

	assert.Equal(t, "exec: \"test command\": file does not exist", result)
}

func TestExecuteCommandResidentBinaryWithMultipleArgs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommandResidentBinary(shortWait, testCommand, testArg, "arg1", "arg2", "arg3")

	assert.NoError(t, err)
}

func TestRunCommandWithOutputTrimSpace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("  test output with spaces  "), nil
	})

	output, err := runCommandWithOutput(&exec.Cmd{}, false)

	assert.NoError(t, err)
	assert.Equal(t, "test output with spaces", output)
}

func TestStartCommandNoEnv(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StdoutPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testOutput))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "StderrPipe", func() (io.ReadCloser, error) {
		return &mockReadCloser{reader: bytes.NewReader([]byte(testError))}, nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})

	result, err := startCommand(context.Background(), []string{}, testCommand, testArg)

	assert.NoError(t, err)
	assert.NotNil(t, result.Cmd)
	assert.NotNil(t, result.Stdout)
	assert.NotNil(t, result.Stderr)
}

func TestCommandExecutorInterface(t *testing.T) {
	var executor Executor = &CommandExecutor{}
	assert.NotNil(t, executor)
}

func TestLogFromReaderEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(log.Debug, func(msg string, v ...interface{}) {})

	reader := &mockReadCloser{reader: strings.NewReader("")}
	logFromReader(reader)
}

func TestRunCommandWithCombinedOutputTrimSpace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("  test output with spaces  "), nil
	})

	output, err := runCommandWithOutput(&exec.Cmd{}, true)

	assert.NoError(t, err)
	assert.Equal(t, "test output with spaces", output)
}

func TestExecuteCommandWithEnvMultipleEnvVars(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testEnv := []string{"VAR1=value1", "VAR2=value2", "VAR3=value3"}
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(startCommand, func(_ context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
		cmd := &exec.Cmd{}
		return StartCommandResult{Cmd: cmd}, nil
	})
	patches.ApplyFunc(logOutput, func(stdout, stderr io.ReadCloser) {})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	err := executor.ExecuteCommandWithEnv(testEnv, testCommand, testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandWithTimeoutWithOutput(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		cmd.Stdout = &bytes.Buffer{}
		cmd.Stderr = &bytes.Buffer{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Start", func() error {
		return nil
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Wait", func() error {
		return nil
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithTimeout(testTimeout, testCommand, testArg)

	assert.NoError(t, err)
	assert.NotNil(t, output)
}

func TestExecuteCommandWithOutputFile(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return nil, nil
	})

	executor := &CommandExecutor{}
	_, err := executor.ExecuteCommandWithOutputFile(testCommand, ">", testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandWithOutputFileWithError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("error output"), os.ErrPermission
	})

	executor := &CommandExecutor{}
	_, err := executor.ExecuteCommandWithOutputFile(testCommand, ">", testArg)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "permission denied")
}

func TestExecuteCommandWithOutputFileTimeout(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.CommandContext, func(ctx context.Context, name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testOutput), nil
	})

	executor := &CommandExecutor{}
	_, err := executor.ExecuteCommandWithOutputFileTimeout(testTimeout, testCommand, ">", testArg)

	assert.NoError(t, err)
}

func TestExecuteCommandWithOutputFileTimeoutWithCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testContent := "error test content"
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
	patches.ApplyFunc(exec.CommandContext, func(ctx context.Context, name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte(testContent), os.ErrPermission
	})

	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithOutputFileTimeout(testTimeout, testCommand, ">", testArg)

	assert.Error(t, err)
	assert.Equal(t, os.ErrPermission, err)
	assert.Equal(t, testContent, output)
}
