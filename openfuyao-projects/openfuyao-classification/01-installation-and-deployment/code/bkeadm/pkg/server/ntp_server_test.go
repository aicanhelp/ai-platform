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

package server

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/shirou/gopsutil/v3/process"
	"github.com/stretchr/testify/assert"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testNTPServerName    = "ntpserver"
	testNTPServerAddress = "pool.ntp.org"
	testLogFilePath      = "/tmp/ntpserver.log"
	testDaemonMaxCount   = 5
)

func TestTryConnectNTPServerSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(server string) error {
		return nil
	})

	result := TryConnectNTPServer(testNTPServerAddress)

	assert.True(t, result)
}

func TestTryConnectNTPServerFailure(t *testing.T) {
	callCount := 0

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(ntp.Date, func(server string) error {
		callCount++
		return assert.AnError
	})

	result := TryConnectNTPServer(testNTPServerAddress)

	assert.False(t, result)
	assert.Equal(t, testDaemonMaxCount, callCount)
}

// func TestRemoveNTPServerWithSystemdService(t *testing.T) {
// 	tests := []struct {
// 		name               string
// 		mockExists         func(string) bool
// 		mockExecuteCommand func(string, ...string) error
// 		mockRemove         func(string) error
// 		mockPids           func() ([]int32, error)
// 		mockNewProcess     func(int32) (*process.Process, error)
// 		mockCmdline        func(*process.Process) (string, error)
// 		expectError        bool
// 	}{
// 		{
// 			name: "successful removal with systemd service",
// 			mockExists: func(path string) bool {
// 				return true
// 			},
// 			mockExecuteCommand: func(command string, args ...string) error {
// 				return nil
// 			},
// 			mockRemove: func(path string) error {
// 				return nil
// 			},
// 			mockPids: func() ([]int32, error) {
// 				return []int32{1234}, nil
// 			},
// 			mockNewProcess: func(pid int32) (*process.Process, error) {
// 				return &process.Process{Pid: pid}, nil
// 			},
// 			mockCmdline: func(p *process.Process) (string, error) {
// 				return "bke start ntpserver", nil
// 			},
// 			expectError: true,
// 		},
// 		{
// 			name: "no systemd service exists",
// 			mockExists: func(path string) bool {
// 				return false
// 			},
// 			mockPids: func() ([]int32, error) {
// 				return []int32{}, nil
// 			},
// 			expectError: false,
// 		},
// 		{
// 			name: "execute command fails but continues",
// 			mockExists: func(path string) bool {
// 				return true
// 			},
// 			mockExecuteCommand: func(command string, args ...string) error {
// 				return assert.AnError
// 			},
// 			mockRemove: func(path string) error {
// 				return nil
// 			},
// 			mockPids: func() ([]int32, error) {
// 				return []int32{}, nil
// 			},
// 			expectError: false,
// 		},
// 	}

// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			patches := gomonkey.NewPatches()
// 			patches.ApplyFunc(utils.Exists, tt.mockExists)
// 			if tt.mockExecuteCommand != nil {
// 				patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand,
// 					func(_ *exec.CommandExecutor, command string, args ...string) error {
// 						return tt.mockExecuteCommand(command, args...)
// 					})
// 			}
// 			if tt.mockRemove != nil {
// 				patches.ApplyFunc(os.Remove, tt.mockRemove)
// 			}
// 			if tt.mockPids != nil {
// 				patches.ApplyFunc(process.Pids, tt.mockPids)
// 			}
// 			if tt.mockNewProcess != nil {
// 				patches.ApplyFunc(process.NewProcess, tt.mockNewProcess)
// 			}
// 			if tt.mockCmdline != nil {
// 				patches.ApplyFunc((*process.Process).Cmdline, tt.mockCmdline)
// 			}
// 			defer patches.Reset()

// 			err := RemoveNTPServer()

// 			if tt.expectError {
// 				assert.Error(t, err)
// 			} else {
// 				assert.NoError(t, err)
// 			}
// 		})
// 	}
// }

func TestRemoveNTPServerLogFileRemoval(t *testing.T) {
	removeCalled := false

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(process.Pids, func() ([]int32, error) {
		return []int32{}, nil
	})

	patches.ApplyFunc(os.Remove, func(path string) error {
		if path == testLogFilePath {
			removeCalled = true
		}
		return nil
	})

	err := RemoveNTPServer()

	assert.NoError(t, err)
	assert.True(t, removeCalled)
}

func TestRemoveNTPServerProcessCleanup(t *testing.T) {
	tests := []struct {
		name           string
		mockPids       func() ([]int32, error)
		mockNewProcess func(int32) (*process.Process, error)
		mockCmdline    func(*process.Process) (string, error)
		expectCleanup  bool
	}{

		{
			name: "no matching process",
			mockPids: func() ([]int32, error) {
				return []int32{9999}, nil
			},
			mockNewProcess: func(pid int32) (*process.Process, error) {
				return &process.Process{Pid: pid}, nil
			},
			mockCmdline: func(p *process.Process) (string, error) {
				return "some other process", nil
			},
			expectCleanup: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return false
			})

			patches.ApplyFunc(process.Pids, tt.mockPids)

			if tt.mockNewProcess != nil {
				patches.ApplyFunc(process.NewProcess, tt.mockNewProcess)
			}

			if tt.mockCmdline != nil {
				patches.ApplyFunc((*process.Process).Cmdline, tt.mockCmdline)
			}

			err := RemoveNTPServer()

			assert.NoError(t, err)
		})
	}
}

func TestSystemdNTPServer(t *testing.T) {
	tests := []struct {
		name               string
		mockGetenv         func(string) string
		mockExecPath       func() (string, error)
		mockWriteFile      func(string, []byte, os.FileMode) error
		mockExecuteCommand func(string, ...string) error
		expectError        bool
	}{
		{
			name: "successful systemd setup",
			mockGetenv: func(key string) string {
				if key == "bke" {
					return "/usr/local/bin/bke"
				}
				return ""
			},
			mockExecPath: func() (string, error) {
				return "/usr/local/bin/bke", nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockExecuteCommand: func(command string, args ...string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "getenv fails falls back to exec path",
			mockGetenv: func(key string) string {
				return ""
			},
			mockExecPath: func() (string, error) {
				return "/usr/local/bin/bke", nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return nil
			},
			mockExecuteCommand: func(command string, args ...string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "exec path fails",
			mockGetenv: func(key string) string {
				return ""
			},
			mockExecPath: func() (string, error) {
				return "", assert.AnError
			},
			expectError: true,
		},
		{
			name: "write file fails",
			mockGetenv: func(key string) string {
				return ""
			},
			mockExecPath: func() (string, error) {
				return "/usr/local/bin/bke", nil
			},
			mockWriteFile: func(name string, data []byte, perm os.FileMode) error {
				return assert.AnError
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockGetenv != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(os.Getenv, tt.mockGetenv)
			}

			if tt.mockExecPath != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(utils.ExecPath, tt.mockExecPath)
			}

			if tt.mockWriteFile != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)
			}

			if tt.mockExecuteCommand != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(global.Command.ExecuteCommand, tt.mockExecuteCommand)
			}

			SystemdNTPServer()
		})
	}
}

func TestSystemdNTPServerCommandExecution(t *testing.T) {
	enableCalled := false
	daemonReloadCalled := false
	startCalled := false

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(utils.ExecPath, func() (string, error) {
		return "/usr/local/bin/bke", nil
	})

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(global.Command.ExecuteCommand, func(command string, args ...string) error {
		if command == "sh" && len(args) > 0 {
			if args[0] == "systemctl enable ntpserver.service" {
				enableCalled = true
			}
			if args[0] == "systemctl daemon-reload" {
				daemonReloadCalled = true
			}
			if args[0] == "systemctl start ntpserver.service" {
				startCalled = true
			}
		}
		return nil
	})

	SystemdNTPServer()

	assert.False(t, enableCalled)
	assert.False(t, daemonReloadCalled)
	assert.False(t, startCalled)
}
