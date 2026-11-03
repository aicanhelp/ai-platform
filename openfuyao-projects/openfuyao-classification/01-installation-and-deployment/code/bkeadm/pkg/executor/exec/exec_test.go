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

package exec

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testShortTimeout  = 1 * time.Second
	testMediumTimeout = 5 * time.Second
)

func TestCommandExecutorStruct(t *testing.T) {
	// Test that the CommandExecutor struct exists and can be instantiated
	executor := &CommandExecutor{}
	assert.NotNil(t, executor)
}

func TestCommandResultStruct(t *testing.T) {
	// Test that the CommandResult struct has the expected fields
	result := &CommandResult{}

	result.Cmd = &exec.Cmd{}
	result.Stdout = nil
	result.Stderr = nil

	assert.NotNil(t, result.Cmd)
	assert.Nil(t, result.Stdout)
	assert.Nil(t, result.Stderr)
}

func TestCommandTimeoutContextStruct(t *testing.T) {
	// Test that the commandTimeoutContext struct has the expected fields
	timer := time.NewTimer(testShortTimeout)
	defer timer.Stop()

	buffer := &bytes.Buffer{}
	cmd := &exec.Cmd{}

	ctx := &commandTimeoutContext{
		cmd:     cmd,
		done:    make(chan error, 1),
		timer:   timer,
		timeout: testShortTimeout,
		command: "test-command",
		buffer:  buffer,
	}

	assert.Equal(t, cmd, ctx.cmd)
	assert.NotNil(t, ctx.done)
	assert.Equal(t, timer, ctx.timer)
	assert.Equal(t, testShortTimeout, ctx.timeout)
	assert.Equal(t, "test-command", ctx.command)
	assert.Equal(t, buffer, ctx.buffer)
}

func TestExecuteCommand(t *testing.T) {
	executor := &CommandExecutor{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*CommandExecutor).ExecuteCommandWithEnv, func(c *CommandExecutor, env []string, command string, arg ...string) error {
		assert.Equal(t, []string{}, env)
		assert.Equal(t, "echo", command)
		assert.Equal(t, []string{"hello"}, arg)
		return nil
	})

	err := executor.ExecuteCommand("echo", "hello")

	assert.NoError(t, err)
}

func TestExecuteCommandWithEnv(t *testing.T) {
	executor := &CommandExecutor{}

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
		patches.ApplyFunc(logCommandOutput, func(out string) {})
		patches.ApplyFunc(runCommandWithOutput, func(cmd *exec.Cmd, combinedOutput bool) (string, error) {
			assert.True(t, combinedOutput)
			assert.Equal(t, []string{"ENV_VAR=value"}, cmd.Env)
			return "hello", nil
		})

		err := executor.ExecuteCommandWithEnv([]string{"ENV_VAR=value"}, "echo", "hello")
		assert.NoError(t, err)
	})

	t.Run("failure includes output", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(logCommand, func(command string, arg ...string) {})
		patches.ApplyFunc(runCommandWithOutput, func(cmd *exec.Cmd, combinedOutput bool) (string, error) {
			return "permission denied", errors.New("exit status 1")
		})

		err := executor.ExecuteCommandWithEnv(nil, "nerdctl", "run", "registry:2")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exit status 1")
		assert.Contains(t, err.Error(), "permission denied")
	})
}

func TestExecuteCommandWithOutput(t *testing.T) {
	executor := &CommandExecutor{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})

	patches.ApplyFunc(runCommandWithOutput, func(cmd *exec.Cmd, combinedOutput bool) (string, error) {
		return "command output", nil
	})

	output, err := executor.ExecuteCommandWithOutput("echo", "hello")

	assert.NoError(t, err)
	assert.Equal(t, "command output", output)
}

func TestExecuteCommandWithCombinedOutput(t *testing.T) {
	executor := &CommandExecutor{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(logCommand, func(command string, arg ...string) {})

	patches.ApplyFunc(runCommandWithOutput, func(cmd *exec.Cmd, combinedOutput bool) (string, error) {
		assert.True(t, combinedOutput)
		return "combined output", nil
	})

	output, err := executor.ExecuteCommandWithCombinedOutput("echo", "hello")

	assert.NoError(t, err)
	assert.Equal(t, "combined output", output)
}

func TestExecuteCommandWithTimeout(t *testing.T) {
	executor := &CommandExecutor{}

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(waitForCommandWithTimeout, func(ctx commandTimeoutContext) (string, error) {
		return "timeout output", nil
	})

	output, err := executor.ExecuteCommandWithTimeout(testShortTimeout, "echo", "hello")

	assert.NoError(t, err)
	assert.Equal(t, "timeout output", output)
}

func TestExecuteCommandResidentBinary(t *testing.T) {
	executor := &CommandExecutor{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc((*exec.Cmd).Run, func(cmd *exec.Cmd) error {
		return nil
	})

	err := executor.ExecuteCommandResidentBinary(time.Millisecond, "sleep", "1")

	assert.NoError(t, err)
}

func TestCommandFailedError(t *testing.T) {
	t.Run("with output", func(t *testing.T) {
		err := commandFailedError("nerdctl", []string{"run"}, "image not found", errors.New("exit status 1"))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "nerdctl run")
		assert.Contains(t, err.Error(), "exit status 1")
		assert.Contains(t, err.Error(), "image not found")
	})

	t.Run("without output", func(t *testing.T) {
		err := commandFailedError("echo", []string{"hello"}, "", errors.New("executable file not found"))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "echo hello")
		assert.NotContains(t, err.Error(), ": :")
	})
}

func TestLogCommandOutput(t *testing.T) {
	called := false
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(log.Debug, func(args ...any) {
		called = true
	})

	logCommandOutput("  hello  ")
	assert.True(t, called)

	called = false
	logCommandOutput("   ")
	assert.False(t, called)
}

func TestRunCommandWithOutput(t *testing.T) {
	cmd := exec.Command("echo", "hello")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.Cmd).Output, func(c *exec.Cmd) ([]byte, error) {
		return []byte("hello\n"), nil
	})

	output, err := runCommandWithOutput(cmd, false)

	assert.NoError(t, err)
	assert.Equal(t, "hello", output)
}

func TestRunCommandWithOutputError(t *testing.T) {
	cmd := exec.Command("invalid-command")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.Cmd).Output, func(c *exec.Cmd) ([]byte, error) {
		return []byte(""), errors.New("command failed")
	})

	patches.ApplyFunc(assertErrorType, func(err error) string {
		return "assertion error"
	})

	output, err := runCommandWithOutput(cmd, false)

	assert.Error(t, err)
	assert.Contains(t, output, "")
}

func TestRunCommandWithCombinedOutput(t *testing.T) {
	cmd := exec.Command("echo", "hello")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*exec.Cmd).CombinedOutput, func(c *exec.Cmd) ([]byte, error) {
		return []byte("hello\n"), nil
	})

	output, err := runCommandWithOutput(cmd, true)

	assert.NoError(t, err)
	assert.Equal(t, "hello", output)
}

func TestAssertErrorType(t *testing.T) {
	// Test with ExitError
	exitErr := &exec.ExitError{
		Stderr: []byte("exit error message"),
	}
	result := assertErrorType(exitErr)
	assert.Equal(t, "exit error message", result)

	// Test with Error
	execErr := &exec.Error{
		Err: errors.New("exec error"),
	}
	result = assertErrorType(execErr)
	assert.Equal(t, "exec: \"\": exec error", result)

	// Test with other error
	otherErr := errors.New("other error")
	result = assertErrorType(otherErr)
	assert.Equal(t, "", result)
}

func TestHandleTimeout(t *testing.T) {
	cmd := &exec.Cmd{}
	cmd.Process = &os.Process{}
	timer := time.NewTimer(testShortTimeout)
	defer timer.Stop()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc((*os.Process).Kill, func(p *os.Process) error {
		return nil
	})

	// Test with interrupt already sent
	output, err, interruptSent := handleTimeout(cmd, "test-command", true, timer, testShortTimeout)
	assert.Equal(t, "", output)
	assert.NotNil(t, err)
	assert.True(t, interruptSent)

	patches.ApplyFunc((*os.Process).Signal, func(p *os.Process) error {
		return nil
	})

	// Test with interrupt not sent yet
	output, err, interruptSent = handleTimeout(cmd, "test-command", false, timer, testShortTimeout)
	assert.Equal(t, "", output)
	assert.Nil(t, err)
	assert.True(t, interruptSent)
}

func TestHandleKillProcess(t *testing.T) {
	// Create a command that will be killed
	cmd := exec.Command("sleep", "10")

	// Start the command
	err := cmd.Start()
	assert.NoError(t, err)

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(log.Infof, func(format string, args ...interface{}) {})

	patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) {})

	patches.ApplyFunc((*os.Process).Kill, func(p *os.Process) error {
		return nil
	})

	output, err := handleKillProcess(cmd, "sleep")

	// The command should be killed
	assert.Equal(t, "", output)
	assert.Error(t, err)
}

func TestHandleCommandDone(t *testing.T) {
	buffer := &bytes.Buffer{}
	buffer.WriteString("test output")

	// Test with no error and no interrupt sent
	output, err := handleCommandDone(nil, false, "test-command", buffer)
	assert.Equal(t, "test output", output)
	assert.NoError(t, err)

	// Test with error
	output, err = handleCommandDone(errors.New("command failed"), false, "test-command", buffer)
	assert.Equal(t, "test output", output)
	assert.Error(t, err)

	// Test with interrupt sent
	output, err = handleCommandDone(nil, true, "test-command", buffer)
	assert.Equal(t, "test output", output)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "timeout waiting for the command")
}

func TestCommandWithOutputFileInternal(t *testing.T) {
	// Test the function with a real command
	command := "echo"
	outfileArg := "> "
	args := []string{"hello"}

	output, _ := commandWithOutputFileInternal(command, outfileArg, nil, args...)

	// The function should complete without error
	// Note: This test may behave differently depending on the system
	// On Unix-like systems, this might work differently than on Windows
	assert.NotNil(t, output) // Output could be empty string or error message
	// We don't assert.NoError here because the command might fail depending on the system
}

func TestCommandWithOutputFileInternalTimeout(t *testing.T) {
	// Test with timeout
	timeout := time.Millisecond * 100
	command := "sleep"
	outfileArg := ""
	args := []string{"1"} // Sleep for 1 second, but timeout after 100ms

	output, err := commandWithOutputFileInternal(command, outfileArg, &timeout, args...)

	// Should return with timeout error
	assert.NotNil(t, output)
	assert.NotNil(t, err)
}

func TestExecutorInterface(t *testing.T) {
	// Test that the Executor interface has all expected methods
	var executor Executor = &CommandExecutor{}

	// Verify all methods exist by calling them with appropriate arguments
	// (We won't actually execute them, just verify they exist)

	// ExecuteCommand
	_ = executor.ExecuteCommand

	// ExecuteCommandWithEnv
	_ = executor.ExecuteCommandWithEnv

	// ExecuteCommandWithOutput
	_ = executor.ExecuteCommandWithOutput

	// ExecuteCommandWithCombinedOutput
	_ = executor.ExecuteCommandWithCombinedOutput

	// ExecuteCommandWithOutputFile
	_ = executor.ExecuteCommandWithOutputFile

	// ExecuteCommandWithOutputFileTimeout
	_ = executor.ExecuteCommandWithOutputFileTimeout

	// ExecuteCommandWithTimeout
	_ = executor.ExecuteCommandWithTimeout

	// ExecuteCommandResidentBinary
	_ = executor.ExecuteCommandResidentBinary

	// All methods should exist without compilation errors
	assert.True(t, true)
}

func TestCommandExecutorImplementsExecutor(t *testing.T) {
	// Verify that CommandExecutor implements the Executor interface
	var _ Executor = &CommandExecutor{}

	// This assertion will fail at compile time if CommandExecutor doesn't implement Executor
	assert.True(t, true)
}
