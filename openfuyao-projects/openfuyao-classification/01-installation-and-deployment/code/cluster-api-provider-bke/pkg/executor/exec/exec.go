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
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// Executor is the main interface for all the exec commands.
// WithContext returns an Executor whose spawned processes are cancelled when ctx is done (P4/A6).
type Executor interface {
	WithContext(ctx context.Context) Executor
	ExecuteCommand(command string, arg ...string) error
	ExecuteCommandWithEnv(env []string, command string, arg ...string) error
	ExecuteCommandWithOutput(command string, arg ...string) (string, error)
	ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error)
	ExecuteCommandWithCombinedOutputInDir(dir string, command string, arg ...string) (string, error)
	ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error)
	ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error)
	ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error)
	ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error
}

// CommandExecutor is the type of the Executor.
// ctx is set via WithContext on a per-invocation copy; the zero value uses context.Background().
type CommandExecutor struct {
	ctx context.Context
}

// Bound returns e.WithContext(ctx). Nil e becomes a fresh CommandExecutor.
func Bound(e Executor, ctx context.Context) Executor {
	if e == nil {
		return (&CommandExecutor{}).WithContext(ctx)
	}
	return e.WithContext(ctx)
}

// WithContext returns a shallow copy bound to ctx for CommandContext-based runs.
func (c *CommandExecutor) WithContext(ctx context.Context) Executor {
	cp := CommandExecutor{}
	if c != nil {
		cp = *c
	}
	if ctx == nil {
		ctx = context.Background()
	}
	cp.ctx = ctx
	return &cp
}

func (c *CommandExecutor) ctxOrBG() context.Context {
	if c != nil && c.ctx != nil {
		return c.ctx
	}
	return context.Background()
}

// ExecuteCommand starts a process and wait for its completion
func (c *CommandExecutor) ExecuteCommand(command string, arg ...string) error {
	return c.ExecuteCommandWithEnv([]string{}, command, arg...)
}

// ExecuteCommandWithEnv starts a process with env variables and wait for its completion
func (c *CommandExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	result, err := startCommand(c.ctxOrBG(), env, command, arg...)
	if err != nil {
		return err
	}

	logOutput(result.Stdout, result.Stderr)

	if err := result.Cmd.Wait(); err != nil {
		return err
	}

	return nil
}

// ExecuteCommandWithTimeout starts a process and waits until timeout or parent ctx cancel (P4/A6).
func (c *CommandExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	parent := c.ctxOrBG()
	ctx := parent
	cancel := func() {
		// timeout<=0 时未创建 WithTimeout ctx，cancel 为空实现，
		// 保持 defer 统一调用以避免两条退出路径的分叉
	}
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(parent, timeout)
	}
	defer cancel()

	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(ctx, command, arg...)
	var b bytes.Buffer
	cmd.Stdout = &b
	cmd.Stderr = &b
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return strings.TrimSpace(b.String()), ctx.Err()
		}
		return strings.TrimSpace(b.String()), err
	}
	return strings.TrimSpace(b.String()), nil
}

// HandleTimeoutParams 包含处理超时所需的参数
type HandleTimeoutParams struct {
	Cmd           *exec.Cmd
	Buffer        *bytes.Buffer
	Command       string
	InterruptSent *bool
}

// handleTimeout 处理命令执行超时的情况
func handleTimeout(params HandleTimeoutParams) (string, error, bool) {
	if *params.InterruptSent {
		log.Infof("timeout waiting for process %s to return after interrupt signal was sent. Sending kill signal to the process", params.Command)
		var e error
		if err := params.Cmd.Process.Kill(); err != nil {
			e = fmt.Errorf("timeout waiting for the command %s to return after interrupt signal was sent; kill process failed: %w", params.Command, err)
		} else {
			e = fmt.Errorf("timeout waiting for the command %s to return", params.Command)
		}
		return strings.TrimSpace(params.Buffer.String()), e, true
	}

	log.Infof("timeout waiting for process %s to return. Sending interrupt signal to the process", params.Command)
	if err := params.Cmd.Process.Signal(os.Interrupt); err != nil {
		log.Debugf("failed to send interrupt signal to process %s: %v", params.Command, err)
		// kill signal will be sent next loop
	}
	*params.InterruptSent = true
	return "", nil, false
}

// HandleCommandCompletionParams 包含处理命令完成所需的参数
type HandleCommandCompletionParams struct {
	Err           error
	Buffer        *bytes.Buffer
	InterruptSent bool
	Command       string
}

// handleCommandCompletion 处理命令完成的情况
func handleCommandCompletion(params HandleCommandCompletionParams) (string, error) {
	if params.Err != nil {
		return strings.TrimSpace(params.Buffer.String()), params.Err
	}
	if params.InterruptSent {
		return strings.TrimSpace(params.Buffer.String()), fmt.Errorf("timeout waiting for the command %s to return", params.Command)
	}
	return strings.TrimSpace(params.Buffer.String()), nil
}

// ExecuteCommandWithOutput executes a command with output
func (c *CommandExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(c.ctxOrBG(), command, arg...)
	return runCommandWithOutput(cmd, false)
}

// ExecuteCommandWithCombinedOutput executes a command with combined output
func (c *CommandExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(c.ctxOrBG(), command, arg...)
	return runCommandWithOutput(cmd, true)
}

// ExecuteCommandWithCombinedOutputInDir executes a command with combined output in the given working directory.
func (c *CommandExecutor) ExecuteCommandWithCombinedOutputInDir(dir string, command string, arg ...string) (string, error) {
	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(c.ctxOrBG(), command, arg...)
	if dir != "" {
		cmd.Dir = dir
	}
	return runCommandWithOutput(cmd, true)
}

// ExecuteCommandWithOutputFileTimeout Same as ExecuteCommandWithOutputFile but with a timeout limit.
// #nosec G307 Calling defer to close the file without checking the error return is not a risk for a simple file open and close
func (c *CommandExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration,
	command, outfileArg string, arg ...string) (string, error) {

	outFile, err := ioutil.TempFile("", "")
	if err != nil {
		return "", fmt.Errorf("failed to open output file: %+w", err)
	}
	defer outFile.Close()
	defer os.Remove(outFile.Name())

	arg = append(arg, outfileArg, outFile.Name())
	logCommand(command, arg...)

	ctx, cancel := context.WithTimeout(c.ctxOrBG(), timeout)
	defer cancel()

	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(ctx, command, arg...)
	cmdOut, err := cmd.CombinedOutput()

	// if there was anything that went to stdout/stderr then log it, even before
	// we return an error
	if string(cmdOut) != "" {
		log.Debug(string(cmdOut))
	}

	if ctx.Err() == context.DeadlineExceeded {
		return string(cmdOut), ctx.Err()
	}

	if err != nil {
		return string(cmdOut), err
	}

	return readOutputFile(outFile)
}

// ExecuteCommandWithOutputFile executes a command with output on a file
// #nosec G307 Calling defer to close the file without checking the error return is not a risk for a simple file open and close
func (c *CommandExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {

	// create a temporary file to serve as the output file for the command to be run and ensure
	// it is cleaned up after this function is done
	outFile, err := ioutil.TempFile("", "")
	if err != nil {
		return "", fmt.Errorf("failed to open output file: %+w", err)
	}
	defer outFile.Close()
	defer os.Remove(outFile.Name())

	// append the output file argument to the list or args
	arg = append(arg, outfileArg, outFile.Name())

	logCommand(command, arg...)
	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(c.ctxOrBG(), command, arg...)
	cmdOut, err := cmd.CombinedOutput()
	if err != nil {
		cmdOut = []byte(fmt.Sprintf("%s. %s", string(cmdOut), assertErrorType(err)))
	}
	// if there was anything that went to stdout/stderr then log it, even before we return an error
	if string(cmdOut) != "" {
		log.Debug(string(cmdOut))
	}
	if err != nil {
		return string(cmdOut), err
	}

	// read the entire output file and return that to the caller
	return readOutputFile(outFile)
}

func (c *CommandExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	ctx := c.ctxOrBG()
	cmd := exec.CommandContext(ctx, command, arg...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGKILL,
	}
	go func() {
		if err := cmd.Run(); err != nil {
			// Resident process runs detached; surface via Debug only (no caller return path).
			log.Debugf("run resident server %s failed: %v", command, err)
		}
	}()
	if timeout <= 0 {
		return nil
	}
	timer := time.NewTimer(timeout)
	select {
	case <-ctx.Done():
		timer.Stop()
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// StartCommandResult 包含 startCommand 函数的执行结果
type StartCommandResult struct {
	Cmd    *exec.Cmd
	Stdout io.ReadCloser
	Stderr io.ReadCloser
}

func startCommand(ctx context.Context, env []string, command string, arg ...string) (StartCommandResult, error) {
	logCommand(command, arg...)
	if ctx == nil {
		ctx = context.Background()
	}

	// #nosec G204 Rook controls the input to the exec arguments
	cmd := exec.CommandContext(ctx, command, arg...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return StartCommandResult{}, fmt.Errorf("open stdout pipe for command %s: %w", command, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return StartCommandResult{}, fmt.Errorf("open stderr pipe for command %s: %w", command, err)
	}

	if len(env) > 0 {
		cmd.Env = env
	}

	if err := cmd.Start(); err != nil {
		return StartCommandResult{}, fmt.Errorf("start command %s: %w", command, err)
	}

	return StartCommandResult{
		Cmd:    cmd,
		Stdout: stdout,
		Stderr: stderr,
	}, nil
}

// read from reader line by line and write it to the log
func logFromReader(reader io.ReadCloser) {
	in := bufio.NewScanner(reader)
	lastLine := ""
	for in.Scan() {
		lastLine = in.Text()
		log.Debug(lastLine)
	}
}

func logOutput(stdout, stderr io.ReadCloser) {
	if stdout == nil || stderr == nil {
		log.Debugf("skip collecting stdout/stderr: nil pipe")
		return
	}
	go logFromReader(stderr)
	logFromReader(stdout)
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

func assertErrorType(err error) string {
	switch errType := err.(type) {
	case *exec.ExitError:
		return string(errType.Stderr)
	case *exec.Error:
		return errType.Error()
	default:
		// For other error types, return the standard error string
		return err.Error()
	}
}

// readOutputFile reads the content from the output file and returns it as a string
func readOutputFile(outFile *os.File) (string, error) {
	fileOut, err := ioutil.ReadAll(outFile)
	if err := outFile.Close(); err != nil {
		return "", err
	}
	return string(fileOut), err
}
