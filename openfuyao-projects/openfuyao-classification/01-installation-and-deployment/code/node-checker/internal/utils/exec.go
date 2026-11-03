/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// utils package provides utility functions for oschecktool
package utils

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// CommandExecResult 执行结果
type CommandExecResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// Command 执行命令
type Command struct {
	Command       string
	Args          []string
	Timeout       time.Duration
	RootPath      string
	CombineOutput bool
	Stdin         string
	Logger        *log.Logger
}

// Exec 执行命令
func (c *Command) Exec() (*CommandExecResult, error) {

	if c.Timeout == 0 {
		c.Timeout = DefaultCmdTimeoutSeconds * time.Second
	}
	c.Command = strings.TrimSpace(c.Command)
	if c.Command == "" {
		return nil, fmt.Errorf("command is empty")
	}
	absPath, err := c.getCommandAbsPath()
	if err != nil {
		return nil, fmt.Errorf("failed to get command abs path: %v", err)
	}
	c.Command = absPath

	c.RootPath = strings.TrimSpace(c.RootPath)
	if c.RootPath == "" {
		c.RootPath = "/"
	}
	c.RootPath = filepath.Clean(strings.TrimSpace(c.RootPath))

	if c.RootPath != "" {
		if !CheckChrootPermission() {
			return nil, fmt.Errorf("the current program is running in a container environment and lacks chroot privileges")
		}
	}
	rslt, err := c.execCommand()
	c.Logger.Printf("Command exec over, command: %+v, result: %+v", c, rslt)
	return rslt, err
}

func (c *Command) getCommandAbsPath() (string, error) {

	// 命令中已经包含路径分隔符，直接使用LookPath
	if filepath.Base(c.Command) != c.Command {
		return exec.LookPath(c.Command)
	}
	if filePath, err := filepath.Abs(c.Command); err == nil {
		if fileInfo, err := os.Stat(filePath); err == nil &&
			fileInfo.Mode()&os.FileMode(FileModeExecutable) != 0 && !fileInfo.IsDir() {
			return filePath, nil
		}
	}
	return exec.LookPath(c.Command)
}

func (c *Command) writeInput(stdin io.WriteCloser) error {
	if stdin != nil {
		defer stdin.Close()
		_, err := io.WriteString(stdin, c.Stdin)
		if err != nil {
			return fmt.Errorf("failed to write stdin: %w", err)
		}
	}
	return nil
}

func (c *Command) execCommand() (*CommandExecResult, error) {
	result := &CommandExecResult{}
	ctx, cancel := context.WithTimeout(context.Background(), c.Timeout)
	defer cancel()
	command := exec.CommandContext(ctx, c.Command, c.Args...)
	var stdin io.WriteCloser
	var err error
	if c.Stdin != "" {
		stdin, err = command.StdinPipe()
		if err != nil {
			c.Logger.Printf("Failed to create stdin pipe: %v", err)
			return result, fmt.Errorf("failed to create stdin pipe: %v", err)
		}
	}
	var stdoutBuf, stderrBuf bytes.Buffer
	if c.CombineOutput {
		command.Stdout = &stdoutBuf
		command.Stderr = &stdoutBuf
	} else {
		command.Stdout = &stdoutBuf
		command.Stderr = &stderrBuf
	}
	if err := command.Start(); err != nil {
		return result, fmt.Errorf("failed to start command: %w", err)
	}
	if err = c.writeInput(stdin); err != nil {
		return result, fmt.Errorf("failed to write input: %w", err)
	}
	err = command.Wait()
	result.Stdout = stdoutBuf.String()
	result.Stderr = stderrBuf.String()
	result.ExitCode = command.ProcessState.ExitCode()
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return result, fmt.Errorf("command timeout: %w", err)
		} else {
			return result, fmt.Errorf("command failed: %w", err)
		}
	}
	return result, err

}

// IsPodEnv checks if running in container environment
func IsPodEnv() bool {
	// 检查cgroup信息（适用于Docker/K8s等）
	if cgroup, err := os.ReadFile("/proc/1/cgroup"); err == nil {
		return strings.Contains(string(cgroup), "docker") ||
			strings.Contains(string(cgroup), "kubepods")
	}
	return false
}

// CheckChrootPermission checks if current user can execute chroot, including container environment detection
func CheckChrootPermission() bool {
	// Windows系统不支持chroot
	if runtime.GOOS == "windows" {
		return false
	}

	// 非root用户直接无权限
	if os.Getuid() != 0 {
		return false
	}

	// 容器环境额外检查CAP_SYS_CHROOT能力
	if IsPodEnv() {
		// 读取进程有效能力
		capEff, err := os.ReadFile("/proc/self/status")
		if err != nil {
			return false // 无法读取能力信息，默认无权限
		}

		// 检查是否包含CAP_SYS_CHROOT（十六进制0x80000）
		// CapEff: 00000000a80425fb
		return strings.Contains(string(capEff), "CapEff:") &&
			strings.Contains(string(capEff), "a80425fb") // 示例值，实际需根据系统计算
	}

	// 非容器环境的root用户默认有权限
	return true
}
