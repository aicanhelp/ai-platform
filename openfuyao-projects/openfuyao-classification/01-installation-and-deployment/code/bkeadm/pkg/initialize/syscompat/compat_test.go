/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package syscompat

import (
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testThreeValue = 3
)

func TestCompat(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(verifyAndInstallIptables, func(platform string) (string, error) {
		return "iptables v1.8.4", nil
	})

	err := Compat()
	assert.NoError(t, err)
}

func TestStopFirewall(t *testing.T) {
	executeCallCount := 0

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
		executeCallCount++
		return nil
	})

	stopFirewall()
	assert.Equal(t, testThreeValue, executeCallCount)
}

func TestSetForwardAccept(t *testing.T) {
	tests := []struct {
		name      string
		mockErr   error
		expectErr bool
	}{
		{
			name:      "success",
			expectErr: false,
		},
		{
			name:      "command failed",
			mockErr:   errors.New("iptables failed"),
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(log.Errorf, func(format string, args ...interface{}) {})

			var called bool
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				called = true
				assert.Equal(t, "iptables", cmd)
				assert.Equal(t, []string{"-P", "FORWARD", "ACCEPT"}, args)
				return tt.mockErr
			})

			err := SetForwardAccept()
			assert.Equal(t, tt.expectErr, err != nil)
			assert.True(t, called)
		})
	}
}

func TestVerifyAndInstallIptables(t *testing.T) {
	tests := []struct {
		name         string
		platform     string
		mockExecOut  func(string, ...string) (string, error)
		mockInstall  func(string) (string, error)
		expectOutput string
		expectErr    bool
	}{
		{
			name:     "iptables already installed",
			platform: "centos",
			mockExecOut: func(cmd string, args ...string) (string, error) {
				return "iptables v1.8.4", nil
			},
			expectOutput: "iptables v1.8.4",
			expectErr:    false,
		},
		{
			name:     "iptables not found",
			platform: "ubuntu",
			mockExecOut: func(cmd string, args ...string) (string, error) {
				return "", errors.New("not found")
			},
			mockInstall: func(platform string) (string, error) {
				return "installed", nil
			},
			expectOutput: "installed",
			expectErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(_ *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				return tt.mockExecOut(cmd, args...)
			})

			if tt.mockInstall != nil {
				patches.ApplyFunc(installIptables, tt.mockInstall)
			}

			output, err := verifyAndInstallIptables(tt.platform)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectOutput, output)
			}
		})
	}
}

func TestInstallIptables(t *testing.T) {
	tests := []struct {
		name      string
		platform  string
		mockYum   func() (string, error)
		mockApt   func() (string, error)
		expectOut string
	}{
		{
			name:      "centos yum",
			platform:  "centos",
			mockYum:   func() (string, error) { return "yum installed", nil },
			expectOut: "yum installed",
		},
		{
			name:      "ubuntu apt",
			platform:  "ubuntu",
			mockApt:   func() (string, error) { return "apt installed", nil },
			expectOut: "apt installed",
		},
		{
			name:      "unsupported",
			platform:  "unknown",
			expectOut: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockYum != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(installIptablesYum, tt.mockYum)
			}
			if tt.mockApt != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(installIptablesApt, tt.mockApt)
			}

			output, err := installIptables(tt.platform)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectOut, output)
		})
	}
}

func TestVerifyIptablesInstallation(t *testing.T) {
	tests := []struct {
		name       string
		mockOutput func() (string, error)
		expectOut  string
		expectErr  bool
	}{
		{
			name: "success",
			mockOutput: func() (string, error) {
				return "iptables v1.8.4", nil
			},
			expectOut: "iptables v1.8.4",
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput, func(_ *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				return tt.mockOutput()
			})

			output, err := verifyIptablesInstallation()
			assert.NoError(t, err)
			assert.Equal(t, tt.expectOut, output)
		})
	}
}

func TestInstallIptablesYum(t *testing.T) {
	tests := []struct {
		name    string
		mockCmd func(string, ...string) error
		mockVer func() (string, error)
	}{
		{
			name: "success",
			mockCmd: func(cmd string, args ...string) error {
				return nil
			},
			mockVer: func() (string, error) { return "iptables v1.8.4", nil },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				return tt.mockCmd(cmd, args...)
			})

			patches.ApplyFunc(verifyIptablesInstallation, tt.mockVer)

			output, err := installIptablesYum()
			assert.NoError(t, err)
			assert.Equal(t, "iptables v1.8.4", output)
		})
	}
}

func TestInstallIptablesApt(t *testing.T) {
	tests := []struct {
		name    string
		mockCmd func(string, ...string) error
	}{
		{
			name: "success",
			mockCmd: func(cmd string, args ...string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				return tt.mockCmd(cmd, args...)
			})

			patches.ApplyFunc(verifyIptablesInstallation, func() (string, error) {
				return "iptables v1.8.4", nil
			})

			output, err := installIptablesApt()
			assert.NoError(t, err)
			assert.Equal(t, "iptables v1.8.4", output)
		})
	}
}

func TestSwitchToLegacyIptables(t *testing.T) {
	tests := []struct {
		name          string
		platform      string
		mockReinstall func() error
		mockUpdateDeb func() error
		mockUpdateFed func() error
		expectError   bool
	}{
		{
			name:     "centos yum",
			platform: "centos",
			mockReinstall: func() error {
				return nil
			},
			expectError: false,
		},
		{
			name:     "ubuntu debian",
			platform: "ubuntu",
			mockUpdateDeb: func() error {
				return nil
			},
			expectError: false,
		},
		{
			name:     "fedora",
			platform: "fedora",
			mockUpdateFed: func() error {
				return nil
			},
			expectError: false,
		},
		{
			name:        "unsupported",
			platform:    "unsupported",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockReinstall != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(reinstallIptablesYum, tt.mockReinstall)
			}
			if tt.mockUpdateDeb != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(updateAlternativesDebian, tt.mockUpdateDeb)
			}
			if tt.mockUpdateFed != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(updateAlternativesFedora, tt.mockUpdateFed)
			}

			err := switchToLegacyIptables(tt.platform)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestReinstallIptablesYum(t *testing.T) {
	tests := []struct {
		name      string
		mockCmd   func(string, ...string) error
		expectErr bool
	}{
		{
			name: "success",
			mockCmd: func(cmd string, args ...string) error {
				return nil
			},
			expectErr: false,
		},
		{
			name: "remove failed",
			mockCmd: func(cmd string, args ...string) error {
				return errors.New("remove failed")
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				return tt.mockCmd(cmd, args...)
			})

			err := reinstallIptablesYum()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUpdateAlternativesDebian(t *testing.T) {
	tests := []struct {
		name    string
		mockCmd func(string, ...string) error
	}{
		{
			name: "success",
			mockCmd: func(cmd string, args ...string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				return tt.mockCmd(cmd, args...)
			})

			err := updateAlternativesDebian()
			assert.NoError(t, err)
		})
	}
}

func TestUpdateAlternativesFedora(t *testing.T) {
	tests := []struct {
		name    string
		mockCmd func(string, ...string) error
	}{
		{
			name: "success",
			mockCmd: func(cmd string, args ...string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommand, func(_ *exec.CommandExecutor, cmd string, args ...string) error {
				return tt.mockCmd(cmd, args...)
			})

			err := updateAlternativesFedora()
			assert.NoError(t, err)
		})
	}
}

func TestRepoUpdate(t *testing.T) {
	tests := []struct {
		name           string
		mockPlatform   func() (string, string, string, error)
		mockExecOutput func(string, ...string) (string, error)
	}{
		{
			name: "ubuntu",
			mockPlatform: func() (string, string, string, error) {
				return "ubuntu", "", "", nil
			},
			mockExecOutput: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
		{
			name: "centos",
			mockPlatform: func() (string, string, string, error) {
				return "centos", "", "", nil
			},
			mockExecOutput: func(cmd string, args ...string) (string, error) {
				return "", nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(host.PlatformInformation, tt.mockPlatform)

			patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithCombinedOutput, func(_ *exec.CommandExecutor, cmd string, args ...string) (string, error) {
				return tt.mockExecOutput(cmd, args...)
			})

			err := RepoUpdate()
			assert.NoError(t, err)
		})
	}
}
