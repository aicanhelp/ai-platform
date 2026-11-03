/*
   Copyright @ 2021 bocloud <fushaosong@beyondcent.com>.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Original file: https://gitee.com/bocloud-open-source/carina/blob/v0.9.1/utils/exec/exec.go
*/

package exec

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// Executor is the main interface for all the exec commands
type Executor interface {
	ExecuteCommand(command string, arg ...string) error
	ExecuteCommandWithEnv(env []string, command string, arg ...string) error
	ExecuteCommandWithOutput(command string, arg ...string) (string, error)
	ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error)
	ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error)
	ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error)
	ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error)
	ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error
}

// CommandExecutor is the type of the Executor
type CommandExecutor struct {
}

// CommandResult holds the result of starting a command.
// Deprecated: kept for backward compatibility with tests; ExecuteCommand no longer uses streaming pipes.
type CommandResult struct {
	Cmd    *exec.Cmd
	Stdout io.ReadCloser
	Stderr io.ReadCloser
}

// commandTimeoutContext holds the context for command timeout handling
type commandTimeoutContext struct {
	cmd     *exec.Cmd
	done    chan error
	timer   *time.Timer
	timeout time.Duration
	command string
	buffer  *bytes.Buffer
}

// ExecuteCommand starts a process and wait for its completion
func (c *CommandExecutor) ExecuteCommand(command string, arg ...string) error {
	return c.ExecuteCommandWithEnv([]string{}, command, arg...)
}

// ExecuteCommandWithEnv starts a process with env variables and wait for its completion.
func (*CommandExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.Command(command, arg...)
	if len(env) > 0 {
		cmd.Env = env
	}

	out, err := runCommandWithOutput(cmd, true)
	if err != nil {
		return commandFailedError(command, arg, out, err)
	}
	logCommandOutput(out)
	return nil
}

// ExecuteCommandWithTimeout starts a process and wait for its completion with timeout.
func (*CommandExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.Command(command, arg...)

	var b bytes.Buffer
	cmd.Stdout = &b
	cmd.Stderr = &b

	if err := cmd.Start(); err != nil {
		return "", err
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	ctx := commandTimeoutContext{
		cmd:     cmd,
		done:    done,
		timer:   timer,
		timeout: timeout,
		command: command,
		buffer:  &b,
	}
	return waitForCommandWithTimeout(ctx)
}

func waitForCommandWithTimeout(ctx commandTimeoutContext) (string, error) {
	interruptSent := false
	for {
		select {
		case <-ctx.timer.C:
			output, err, newInterruptSent := handleTimeout(ctx.cmd, ctx.command, interruptSent, ctx.timer, ctx.timeout)
			interruptSent = newInterruptSent
			if err != nil {
				return strings.TrimSpace(ctx.buffer.String()), err
			}
			if output != "" {
				return output, err
			}
		case err := <-ctx.done:
			return handleCommandDone(err, interruptSent, ctx.command, ctx.buffer)
		}
	}
}

func handleTimeout(cmd *exec.Cmd, command string, interruptSent bool, timer *time.Timer,
	timeout time.Duration) (string, error, bool) {
	if interruptSent {
		output, err := handleKillProcess(cmd, command)
		return output, err, interruptSent
	}
	output, err, newInterruptSent := handleInterruptProcess(cmd, command, timer, timeout)
	return output, err, newInterruptSent
}

func handleKillProcess(cmd *exec.Cmd, command string) (string, error) {
	log.Infof("timeout waiting for process %s to return after interrupt signal was sent. "+
		"Sending kill signal to the process", command)
	if err := cmd.Process.Kill(); err != nil {
		log.Errorf("Failed to kill process %s: %v", command, err)
		return "", fmt.Errorf("timeout waiting for the command %s to return after interrupt signal was sent. "+
			"Tried to kill the process but that failed: %v", command, err)
	}
	return "", fmt.Errorf("timeout waiting for the command %s to return", command)
}

func handleInterruptProcess(cmd *exec.Cmd, command string, timer *time.Timer,
	timeout time.Duration) (string, error, bool) {
	log.Infof("timeout waiting for process %s to return. Sending interrupt signal to the process", command)
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		log.Errorf("Failed to send interrupt signal to process %s: %v", command, err)
	}
	timer.Reset(timeout)
	return "", nil, true
}

func handleCommandDone(err error, interruptSent bool, command string, b *bytes.Buffer) (string, error) {
	if err != nil {
		return strings.TrimSpace(b.String()), err
	}
	if interruptSent {
		return strings.TrimSpace(b.String()), fmt.Errorf("timeout waiting for the command %s to return", command)
	}
	return strings.TrimSpace(b.String()), nil
}

// ExecuteCommandWithOutput executes a command with output
func (*CommandExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.Command(command, arg...)
	return runCommandWithOutput(cmd, false)
}

// ExecuteCommandWithCombinedOutput executes a command with combined output
func (*CommandExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.Command(command, arg...)
	return runCommandWithOutput(cmd, true)
}

// ExecuteCommandWithOutputFileTimeout Same as ExecuteCommandWithOutputFile but with a timeout limit.
// #nosec G307 Calling defer to close the file without checking the error return is not a risk for a simple file open and close
func (*CommandExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration,
	command, outfileArg string, arg ...string) (string, error) {
	return commandWithOutputFileInternal(command, outfileArg, &timeout, arg...)
}

// ExecuteCommandWithOutputFile executes a command with output on a file
// #nosec G307 Calling defer to close the file without checking the error return is not a risk for a simple file open and close
func (*CommandExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return commandWithOutputFileInternal(command, outfileArg, nil, arg...)
}

// commandWithOutputFileInternal is the common implementation for executing commands with output files
// If timeout is nil, the command runs without a timeout
func commandWithOutputFileInternal(command, outfileArg string,
	timeout *time.Duration, arg ...string) (string, error) {
	outFile, err := os.CreateTemp("", "")
	if err != nil {
		return "", fmt.Errorf("failed to open output file: %+v", err)
	}
	defer outFile.Close()
	defer os.Remove(outFile.Name())

	arg = append(arg, outfileArg, outFile.Name())
	logCommand(command, arg...)

	var ctx context.Context
	var cancel context.CancelFunc
	var cmd *exec.Cmd

	if timeout != nil {
		ctx, cancel = context.WithTimeout(context.Background(), *timeout)
		defer cancel()
		// #nosec G204 Rook controls the input to the exec arguments
		cmd = exec.CommandContext(ctx, command, arg...)
	} else {
		// #nosec G204 Rook controls the input to the exec arguments
		cmd = exec.Command(command, arg...)
	}

	if cmd == nil {
		return "", fmt.Errorf("failed to create command")
	}

	cmdOut, err := cmd.CombinedOutput()

	// Check for timeout error
	if timeout != nil && ctx.Err() == context.DeadlineExceeded {
		return string(cmdOut), ctx.Err()
	}

	if err != nil {
		if timeout == nil {
			cmdOut = []byte(fmt.Sprintf("%s. %s", string(cmdOut), assertErrorType(err)))
		}
		return string(cmdOut), err
	}

	logCommandOutput(string(cmdOut))

	fileOut, err := io.ReadAll(outFile)
	if err := outFile.Close(); err != nil {
		return "", err
	}
	return string(fileOut), err
}

func (*CommandExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	cmd := exec.Command(command, arg...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGKILL,
	}
	go func() {
		if err := cmd.Run(); err != nil {
			log.Errorf("run Resident server failed: %s+v", err)
		}
	}()
	time.Sleep(timeout)
	return nil
}

func runCommandWithOutput(cmd *exec.Cmd, combinedOutput bool) (string, error) {
	var output []byte
	var err error
	var out string

	if combinedOutput {
		output, err = cmd.CombinedOutput()
	} else {
		output, err = cmd.Output()
		if err != nil {
			output = []byte(fmt.Sprintf("%s. %s", string(output), assertErrorType(err)))
		}
	}

	out = strings.TrimSpace(string(output))

	if err != nil {
		return out, err
	}

	return out, nil
}

func logCommand(command string, arg ...string) {
	log.Debugf("Running command: %s %s", command, strings.Join(arg, " "))
}

func logCommandOutput(out string) {
	out = strings.TrimSpace(out)
	if out != "" {
		log.Debug(out)
	}
}

func commandFailedError(command string, arg []string, out string, err error) error {
	out = strings.TrimSpace(out)
	if out == "" {
		return fmt.Errorf("%s %s: %w", command, strings.Join(arg, " "), err)
	}
	return fmt.Errorf("%s %s: %w: %s", command, strings.Join(arg, " "), err, out)
}

func assertErrorType(err error) string {
	switch errType := err.(type) {
	case *exec.ExitError:
		return string(errType.Stderr)
	case *exec.Error:
		return errType.Error()
	default:
		return ""
	}
}
